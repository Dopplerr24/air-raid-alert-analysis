"""
model.py
========
Walk-forward splitter, heuristic baselines and the LightGBM classifier, for a
panel indexed by (timestamp, region).

Everything that LEARNS from data lives here, behind one small interface:

    model.fit(X_train, y_train)      # sees the training fold only
    p = model.predict_proba(X_test)  # 1-D array of P(onset)

Leakage rules enforced in this module
-------------------------------------
* `walk_forward_splits` cuts on TIME, with one cut shared by every region.
  A per-region or random split would be invalid for a panel: the national
  features of a training row in region A carry information about the same
  hour in region B, so that hour must never sit in the test set.
* Every training timestamp is at least `embargo_hours` before the first test
  timestamp. The window expands; it never shuffles.
* Each model is re-created and re-fitted per fold. Base rates, climatology
  tables, the handling of the `region` category and the trees are all
  estimated from the training fold passed to `fit` and from nothing else.
* Hyper-parameters are fixed constants, not tuned against the test folds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd

LOCAL_TZ = "Europe/Kyiv"


# --------------------------------------------------------------------------- #
# 1. Walk-forward validation with an embargo
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Fold:
    number: int
    train_mask: np.ndarray       # boolean, one entry per row of the panel
    test_mask: np.ndarray
    train_end: pd.Timestamp      # last training timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp       # last test timestamp


def walk_forward_splits(timestamps: pd.DatetimeIndex, n_splits: int = 5,
                        test_days: int = 90, embargo_hours: int = 24,
                        horizon: int = 1) -> Iterator[Fold]:
    """Expanding-window walk-forward splits on the time axis of a panel.

        |------------- train -------------|~embargo~|--- test 1 ---|
        |------------------- train -------------------|~embargo~|--- test 2 ---|

    `timestamps` is the timestamp of every panel row (values repeat, once per
    region). The last `n_splits * test_days` days are cut into consecutive test
    blocks. For a block starting at T, training rows are those with

        t < T - embargo_hours            (in every region alike)

    Why an embargo, given causal features:
      * A row's label looks `horizon` hours ahead, so the last training labels
        would otherwise overlap the first test hours. embargo >= horizon purges
        that overlap (enforced below).
      * Attacks arrive in multi-hour, multi-region episodes. Dropping a full
        day stops one episode from sitting on both sides of the boundary.
    Test features still look back into the training period (lags, 7-day
    windows). That is legitimate: at forecast time that history is known.
    """
    if embargo_hours < horizon:
        raise ValueError(f"embargo_hours ({embargo_hours}) must be >= horizon ({horizon})")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("panel rows must be sorted by timestamp")

    test_len = pd.Timedelta(days=test_days)
    embargo = pd.Timedelta(hours=embargo_hours)
    end = timestamps[-1] + pd.Timedelta(hours=1)          # exclusive end of data
    first_test_start = end - n_splits * test_len

    for k in range(n_splits):
        test_start = first_test_start + k * test_len
        train_mask = np.asarray(timestamps < test_start - embargo)
        test_mask = np.asarray((timestamps >= test_start) & (timestamps < test_start + test_len))
        if not train_mask.any() or not test_mask.any():
            raise ValueError("Not enough history for the requested splits")

        train_end = timestamps[train_mask][-1]
        # Hard guarantees, checked on every fold.
        assert train_end < timestamps[test_mask][0] - embargo, "embargo violated"
        assert not (train_mask & test_mask).any(), "train/test overlap"
        yield Fold(k + 1, train_mask, test_mask, train_end,
                   timestamps[test_mask][0], timestamps[test_mask][-1])


# --------------------------------------------------------------------------- #
# 2. Heuristic baselines
# --------------------------------------------------------------------------- #
def _region(X: pd.DataFrame) -> np.ndarray:
    return X["region"].astype(str).to_numpy()


class GlobalBaseRate:
    """One number for the whole country: the training-fold onset rate."""
    name = "Global base rate"

    def fit(self, X, y):
        self.rate_ = float(y.mean())
        return self

    def predict_proba(self, X):
        return np.full(len(X), self.rate_)


class RegionBaseRate:
    """Each region's own training-fold onset rate.

    This is the honest no-skill reference for a panel: pooled metrics reward
    any model that merely knows Sumska is alerted more often than Zakarpatska.
    """
    name = "Region base rate"

    def fit(self, X, y):
        self.base_ = float(y.mean())
        self.rates_ = pd.Series(y.to_numpy(dtype=float)).groupby(_region(X)).mean().to_dict()
        return self

    def predict_proba(self, X):
        return pd.Series(_region(X)).map(self.rates_).fillna(self.base_).to_numpy(dtype=float)


class PersistenceModel:
    """'What is happening now continues': score = active_lag1.

    raw=True   output the 0/1 state itself. Nothing is fitted.
    raw=False  training-fold onset rate per (region, state).
    """

    def __init__(self, raw: bool = True):
        self.raw = raw
        self.name = "Persistence (raw 0/1)" if raw else "Persistence (fitted rate)"

    def fit(self, X, y):
        self.base_ = float(y.mean())
        keys = pd.MultiIndex.from_arrays([_region(X), X["active_lag1"].astype(int).to_numpy()])
        self.rates_ = pd.Series(y.to_numpy(dtype=float), index=keys).groupby(level=[0, 1]).mean()
        return self

    def predict_proba(self, X):
        state = X["active_lag1"].astype(int).to_numpy()
        if self.raw:
            return state.astype(float)
        keys = pd.MultiIndex.from_arrays([_region(X), state])
        return self.rates_.reindex(keys).fillna(self.base_).to_numpy(dtype=float)


class HourOfWeekClimatology:
    """Mean onset rate per (region, day-of-week, hour) in Kyiv local time.

    The table (168 cells per region) is built in `fit` from the training fold
    only. `shrinkage` pulls thin cells toward that region's base rate:
        rate = (sum_y + m * region_rate) / (n + m)
    """
    name = "Hour-of-week climatology"

    def __init__(self, shrinkage: float = 20.0):
        self.shrinkage = shrinkage

    @staticmethod
    def _cell(X: pd.DataFrame) -> np.ndarray:
        local = X.index.get_level_values("timestamp").tz_convert(LOCAL_TZ)
        return np.asarray(local.dayofweek * 24 + local.hour)

    def fit(self, X, y):
        self.base_ = float(y.mean())
        df = pd.DataFrame({"r": _region(X), "c": self._cell(X), "y": y.to_numpy(dtype=float)})
        region_rate = df.groupby("r")["y"].mean()
        grp = df.groupby(["r", "c"])["y"].agg(["sum", "count"])
        prior = grp.index.get_level_values("r").map(region_rate).to_numpy()
        m = self.shrinkage
        self.table_ = (grp["sum"] + m * prior) / (grp["count"] + m)
        self.region_rate_ = region_rate.to_dict()
        return self

    def predict_proba(self, X):
        r = _region(X)
        keys = pd.MultiIndex.from_arrays([r, self._cell(X)])
        fallback = pd.Series(r).map(self.region_rate_).fillna(self.base_).to_numpy(dtype=float)
        p = self.table_.reindex(keys).to_numpy(dtype=float)
        return np.where(np.isnan(p), fallback, p)


# --------------------------------------------------------------------------- #
# 3. LightGBM
# --------------------------------------------------------------------------- #
# Deliberately restrained: rows are strongly autocorrelated in time and across
# regions, so the effective sample is far smaller than the row count.
LGBM_PARAMS = dict(
    objective="binary",
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=6,
    min_child_samples=500,     # each leaf must cover >= 500 region-hours
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    cat_smooth=50.0,           # regularise splits on the region category
    random_state=42,
    n_jobs=4,
    verbose=-1,
)


@dataclass
class LGBMOnsetModel:
    """LightGBM binary classifier for alert onset.

    `region` is passed as a pandas categorical and handled natively by
    LightGBM (no one-hot, no target encoding of our own). The category
    statistics LightGBM uses to order regions are computed inside `fit`,
    i.e. on the training fold only.

    exclude
        Column-name prefixes to hide from the model. Used for ablations, e.g.
        exclude=("national_",) trains on each region's own history only.

    imbalance / prior_correction
        "none" (default): plain log-loss on the natural class balance. In the
            single-region study this ranked slightly better than weighting and
            needs no probability repair.
        "weighted": scale_pos_weight = n_neg / n_pos from the training fold.
            Raw outputs are then inflated; prior_correction=True undoes it
            analytically with  p = p_w / (p_w + w * (1 - p_w)).
    """
    name: str = "LightGBM"
    exclude: tuple[str, ...] = ()
    imbalance: str = "none"
    prior_correction: bool = True
    params: dict = field(default_factory=lambda: dict(LGBM_PARAMS))

    def __post_init__(self):
        if self.imbalance not in ("weighted", "none"):
            raise ValueError("imbalance must be 'weighted' or 'none'")

    def _columns(self, X: pd.DataFrame) -> list[str]:
        return [c for c in X.columns if not (self.exclude and c.startswith(self.exclude))]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LGBMOnsetModel":
        from lightgbm import LGBMClassifier

        n_pos = int(y.sum())
        if n_pos == 0:
            raise ValueError("training fold has no positive examples")
        self.pos_weight_ = (len(y) - n_pos) / n_pos if self.imbalance == "weighted" else 1.0

        self.feature_names_ = self._columns(X)
        self.model_ = LGBMClassifier(**self.params, scale_pos_weight=self.pos_weight_)
        # pandas `category` dtype columns are picked up as categorical features.
        self.model_.fit(X[self.feature_names_], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.model_.predict_proba(X[self.feature_names_])[:, 1]
        if self.imbalance == "weighted" and self.prior_correction:
            w = self.pos_weight_
            p = p / (p + w * (1.0 - p))
        return p

    def feature_importance(self) -> pd.Series:
        """Share of total gain per feature (training fold of the fitted model)."""
        gain = self.model_.booster_.feature_importance(importance_type="gain")
        return pd.Series(gain / gain.sum(), index=self.feature_names_).sort_values(ascending=False)


PRIMARY_MODEL = "LightGBM global"


def make_models() -> list:
    """Fresh, unfitted instances. Call once per fold so no state carries over."""
    return [
        GlobalBaseRate(),
        RegionBaseRate(),
        PersistenceModel(raw=True),
        PersistenceModel(raw=False),
        HourOfWeekClimatology(),
        # Ablation: same model without the cross-region block.
        LGBMOnsetModel(name="LightGBM own history", exclude=("national_", "neighbors_")),
        LGBMOnsetModel(name=PRIMARY_MODEL),
    ]
