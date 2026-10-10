"""
evaluate.py
===========
Walk-forward evaluation on the regional panel.

Metrics (per fold, then averaged over folds)
--------------------------------------------
pr_auc        Average precision over all region-hours of the fold (pooled).
              PRIMARY metric.
pr_auc_macro  Average precision computed inside each region, then averaged.
              Pooled PR-AUC rewards a model for simply knowing which regions
              are alerted most; the macro figure removes that and measures
              timing skill only. Read both.
brier         Mean squared error of the probability. Lower is better.
bss           Brier skill score vs "Region base rate" in the same fold:
              1 - brier / brier_reference. > 0 beats per-region climatology.
mean_pred     Average predicted probability; compare with `prevalence` to
              spot over- or under-forecasting.

ROC-AUC is intentionally not reported: with ~10% positives it is dominated by
the easy negatives and flatters models that are useless in practice.

Leakage notes
-------------
* The test fold is touched exactly once per model, by `predict_proba`.
* Metrics are computed per fold and then averaged. Pooling predictions across
  folds would mix probabilities from models trained on different histories.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from model import LGBMOnsetModel, make_models, walk_forward_splits

logger = logging.getLogger(__name__)

REFERENCE_MODEL = "Region base rate"


def score(y_true: np.ndarray, p: np.ndarray, regions: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """Metrics for one model on one test fold: (pooled dict, per-region frame)."""
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("predictions must be probabilities in [0, 1]")

    df = pd.DataFrame({"y": y_true, "p": p, "region": regions})
    rows = []
    for region, g in df.groupby("region"):
        rows.append({
            "region": region,
            "prevalence": g["y"].mean(),
            # Average precision is undefined without a positive example.
            "pr_auc": average_precision_score(g["y"], g["p"]) if g["y"].any() else np.nan,
            "brier": brier_score_loss(g["y"], g["p"]),
            "mean_pred": g["p"].mean(),
        })
    by_region = pd.DataFrame(rows)

    pooled = {
        "prevalence": float(np.mean(y_true)),
        "pr_auc": average_precision_score(y_true, p),
        "pr_auc_macro": float(by_region["pr_auc"].mean()),   # NaN regions skipped
        "brier": brier_score_loss(y_true, p),
        "mean_pred": float(np.mean(p)),
    }
    return pooled, by_region


def cross_validate(X: pd.DataFrame, y: pd.Series, model_factory=make_models,
                   n_splits: int = 5, test_days: int = 90, embargo_hours: int = 24,
                   horizon: int = 1) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run every model from `model_factory()` through every fold.

    Returns
    -------
    results    one row per (fold, model) with pooled metrics and `bss`
    by_region  one row per (fold, model, region)
    importance {model name: gain share on the final fold} for LightGBM models
    """
    assert X.index.equals(y.index), "X and y are misaligned"
    timestamps = X.index.get_level_values("timestamp")
    regions = X["region"].astype(str).to_numpy()
    rows, region_rows, importance = [], [], {}

    for fold in walk_forward_splits(timestamps, n_splits, test_days, embargo_hours, horizon):
        X_tr, y_tr = X[fold.train_mask], y[fold.train_mask]
        X_te, y_te = X[fold.test_mask], y[fold.test_mask]
        logger.info("fold %d  train -> %s (%d rows)  |  test %s -> %s (%d rows, %d onsets)",
                    fold.number, fold.train_end.date(), len(X_tr), fold.test_start.date(),
                    fold.test_end.date(), len(X_te), int(y_te.sum()))

        for model in model_factory():          # fresh instances: no state survives a fold
            model.fit(X_tr, y_tr)              # training fold only
            p = model.predict_proba(X_te)      # the single touch of the test fold
            pooled, per_region = score(y_te.to_numpy(), p, regions[fold.test_mask])
            rows.append({"fold": fold.number, "model": model.name,
                         "test_start": fold.test_start.date(), **pooled})
            region_rows.append(per_region.assign(fold=fold.number, model=model.name))
            if isinstance(model, LGBMOnsetModel):
                importance[model.name] = model.feature_importance()

    results = pd.DataFrame(rows)
    ref = results.loc[results["model"] == REFERENCE_MODEL].set_index("fold")["brier"]
    if not ref.empty:
        results["bss"] = 1.0 - results["brier"] / results["fold"].map(ref)
    return results, pd.concat(region_rows, ignore_index=True), importance


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Mean across folds (and std of the primary metric), in run order."""
    order = results["model"].drop_duplicates().tolist()
    agg = results.groupby("model").agg(
        pr_auc=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
        pr_auc_macro=("pr_auc_macro", "mean"),
        brier=("brier", "mean"), bss=("bss", "mean"), mean_pred=("mean_pred", "mean"),
    )
    return agg.loc[order]


def per_fold_table(results: pd.DataFrame, metric: str = "pr_auc") -> pd.DataFrame:
    """Models x folds for one metric, with the fold prevalence on top."""
    table = results.pivot(index="model", columns="fold", values=metric)
    table = table.loc[results["model"].drop_duplicates()]
    prevalence = results.groupby("fold")["prevalence"].first().rename("prevalence").to_frame().T
    return pd.concat([prevalence, table])


def per_region_table(by_region: pd.DataFrame, model: str, reference: str = REFERENCE_MODEL
                     ) -> pd.DataFrame:
    """Per-region means over folds for one model, with its lift over the
    region's own prevalence (the no-skill PR-AUC inside a region)."""
    def mean_of(name):
        return by_region[by_region["model"] == name].groupby("region")[
            ["prevalence", "pr_auc", "brier", "mean_pred"]].mean()

    m, r = mean_of(model), mean_of(reference)
    out = m.assign(pr_auc_lift=m["pr_auc"] / m["prevalence"],
                   bss=1.0 - m["brier"] / r["brier"],
                   pred_over_actual=m["mean_pred"] / m["prevalence"])
    return out.sort_values("pr_auc_lift", ascending=False)
