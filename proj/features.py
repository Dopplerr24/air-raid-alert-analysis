"""
features.py
===========
Strictly causal feature engineering for the alert-onset target, on a panel of
regions.

Temporal alignment (read this first)
------------------------------------
Input: {region: hourly frame} from data_loader. In every frame, row `t`
summarises the bin [t, t+1h) and is only fully known at t+1h. All frames share
one hourly index.

A model row (t, region) is a forecast issued AT instant `t` for that region:

      known at t                  |  to be predicted
      bins ... t-2h, t-1h         |  bin [t, t+1h)
      ----------------------------+---------------------> time
                                  t

    target   y[t, r] = alert_onset_r[t]         (an alert starts in [t, t+1h))
    features X[t, r] = f(bins strictly before t, any region) via shift(k), k >= 1

The newest value a feature may touch is `shift(1)` = bin [t-1h, t), which
closes exactly at the forecast instant. Bin `t` of ANY region is the future.

Two ways a panel can leak, and how each is closed
-------------------------------------------------
1. Look-ahead in time.
   * Every history value enters through `_past()`, which rejects lag < 1.
   * Rolling windows are applied AFTER the shift.
   * `check_causality()` scrambles every region's data from a cut-off onward
     and asserts that no feature at or before the cut-off changes.

2. Bleed between regions (the classic long-format bug: `df.shift(1)` on a
   stacked table hands the last row of region A to the first row of region B).
   * It is impossible by construction here: per-region features are computed
     on that region's OWN frame, one frame at a time, each with its own
     DatetimeIndex. There is no stacked table to shift. Regions are stacked
     only at the very end, after all shifting and rolling is finished.
   * Cross-region information is allowed in exactly one place, the national
     and neighbour aggregates, computed on a timestamp x region matrix with
     `axis=1` sums of already-shifted columns.
   * `check_region_isolation()` rewrites one region's entire history and
     asserts that the own-history features of all other regions are unchanged.

Nothing in this module is fitted (no scalers, encoders or imputers). `region`
is a pandas categorical whose categories are the fixed list of region names,
not statistics of the data. Anything that learns belongs in model.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOCAL_TZ = "Europe/Kyiv"

SEASONAL_LAGS = (24, 168)      # same hour yesterday / same hour last week
ROLLING_WINDOWS = (24, 168)    # onset counts over the past day / week
SPATIAL_ROLL = 3               # national / neighbour onsets over the past 3 hours

OWN_PREFIXES = ("active_", "minutes_", "onset_", "onsets_")
SPATIAL_PREFIXES = ("national_", "neighbors_")

# Land borders between regions. Used only by the optional neighbour features.
# Luhanska oblast and Crimea are absent (no usable data), so they never count.
ADJACENCY: dict[str, tuple[str, ...]] = {
    "Cherkaska oblast": ("Kyivska oblast", "Poltavska oblast", "Kirovohradska oblast", "Vinnytska oblast"),
    "Chernihivska oblast": ("Kyivska oblast", "Sumska oblast", "Poltavska oblast"),
    "Chernivetska oblast": ("Ivano-Frankivska oblast", "Ternopilska oblast", "Khmelnytska oblast", "Vinnytska oblast"),
    "Dnipropetrovska oblast": ("Poltavska oblast", "Kharkivska oblast", "Donetska oblast", "Zaporizka oblast",
                               "Khersonska oblast", "Mykolaivska oblast", "Kirovohradska oblast"),
    "Donetska oblast": ("Kharkivska oblast", "Dnipropetrovska oblast", "Zaporizka oblast"),
    "Ivano-Frankivska oblast": ("Lvivska oblast", "Ternopilska oblast", "Chernivetska oblast", "Zakarpatska oblast"),
    "Kharkivska oblast": ("Sumska oblast", "Poltavska oblast", "Dnipropetrovska oblast", "Donetska oblast"),
    "Khersonska oblast": ("Mykolaivska oblast", "Dnipropetrovska oblast", "Zaporizka oblast"),
    "Khmelnytska oblast": ("Rivnenska oblast", "Zhytomyrska oblast", "Vinnytska oblast", "Chernivetska oblast",
                           "Ternopilska oblast"),
    "Kirovohradska oblast": ("Cherkaska oblast", "Poltavska oblast", "Dnipropetrovska oblast",
                             "Mykolaivska oblast", "Odeska oblast", "Vinnytska oblast"),
    "Kyiv City": ("Kyivska oblast",),
    "Kyivska oblast": ("Kyiv City", "Zhytomyrska oblast", "Vinnytska oblast", "Cherkaska oblast",
                       "Poltavska oblast", "Chernihivska oblast"),
    "Lvivska oblast": ("Volynska oblast", "Rivnenska oblast", "Ternopilska oblast", "Ivano-Frankivska oblast",
                       "Zakarpatska oblast"),
    "Mykolaivska oblast": ("Odeska oblast", "Kirovohradska oblast", "Dnipropetrovska oblast", "Khersonska oblast"),
    "Odeska oblast": ("Vinnytska oblast", "Kirovohradska oblast", "Mykolaivska oblast"),
    "Poltavska oblast": ("Kyivska oblast", "Chernihivska oblast", "Sumska oblast", "Kharkivska oblast",
                         "Dnipropetrovska oblast", "Kirovohradska oblast", "Cherkaska oblast"),
    "Rivnenska oblast": ("Volynska oblast", "Lvivska oblast", "Ternopilska oblast", "Khmelnytska oblast",
                         "Zhytomyrska oblast"),
    "Sumska oblast": ("Chernihivska oblast", "Poltavska oblast", "Kharkivska oblast"),
    "Ternopilska oblast": ("Lvivska oblast", "Rivnenska oblast", "Khmelnytska oblast", "Chernivetska oblast",
                           "Ivano-Frankivska oblast"),
    "Vinnytska oblast": ("Zhytomyrska oblast", "Kyivska oblast", "Cherkaska oblast", "Kirovohradska oblast",
                         "Odeska oblast", "Chernivetska oblast", "Khmelnytska oblast"),
    "Volynska oblast": ("Rivnenska oblast", "Lvivska oblast"),
    "Zakarpatska oblast": ("Lvivska oblast", "Ivano-Frankivska oblast"),
    "Zaporizka oblast": ("Khersonska oblast", "Dnipropetrovska oblast", "Donetska oblast"),
    "Zhytomyrska oblast": ("Rivnenska oblast", "Khmelnytska oblast", "Vinnytska oblast", "Kyivska oblast"),
}
assert all(a in ADJACENCY[b] for a, nbrs in ADJACENCY.items() for b in nbrs), "ADJACENCY must be symmetric"


# --------------------------------------------------------------------------- #
# Causal gateway and target
# --------------------------------------------------------------------------- #
def _past(data, lag: int):
    """The ONLY gateway from raw hourly data into a feature.

    Accepts a Series (one region) or a timestamp x region DataFrame. In both
    cases the index is time only, so shift(lag) moves values `lag` hours down
    the time axis and can never cross from one region into another.
    lag >= 1 guarantees row t never sees its own bin.
    """
    if lag < 1:
        raise ValueError(f"lag must be >= 1 to stay causal, got {lag}")
    if isinstance(data.index, pd.MultiIndex):
        raise TypeError("_past() refuses stacked (multi-index) data: shift it per region instead")
    return data.shift(lag)


def make_target(hourly: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """y[t] = 1 if an alert starts in this region within (t, t + horizon hours].

    The single place where the future is read on purpose.
    horizon=1 -> alert_onset[t] itself. horizon=H -> max over bins t .. t+H-1
    via shift(-k); the last H-1 rows have an incomplete window and become NaN.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    onset = hourly["alert_onset"].astype("float64")
    ahead = pd.concat([onset.shift(-k) for k in range(horizon)], axis=1)
    return ahead.max(axis=1, skipna=False).rename("y")


# --------------------------------------------------------------------------- #
# Feature blocks
# --------------------------------------------------------------------------- #
def make_own_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """History of ONE region, computed on that region's own frame.

        active_lag1         alert already running at the forecast instant t
        minutes_lag1        minutes of alert during the last completed hour
        onset_lag{24,168}   alert started in the same hour yesterday / last week
        onsets_roll{24,168} alert starts in the previous 24 h / 7 days
    """
    X = pd.DataFrame(index=hourly.index)
    X["active_lag1"] = _past(hourly["active_at_end"], 1)
    X["minutes_lag1"] = _past(hourly["alert_minutes"], 1)
    for lag in SEASONAL_LAGS:
        X[f"onset_lag{lag}"] = _past(hourly["alert_onset"], lag)

    # Shift FIRST, then roll: at row t the window covers bins t-w .. t-1.
    # min_periods=w leaves warm-up rows NaN instead of using a short window.
    onsets_known = _past(hourly["n_onsets"], 1)
    for w in ROLLING_WINDOWS:
        X[f"onsets_roll{w}"] = onsets_known.rolling(window=w, min_periods=w).sum()
    return X


def make_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar of the TARGET bin in Kyiv local time. Deterministic and known
    arbitrarily far ahead, so it cannot leak. Local time rather than UTC keeps
    the daily rhythm fixed across daylight-saving changes."""
    local = index.tz_convert(LOCAL_TZ)
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * local.hour / 24),
            "hour_cos": np.cos(2 * np.pi * local.hour / 24),
            "dow": local.dayofweek,
        },
        index=index,
    )


def _wide(frames: dict[str, pd.DataFrame], column: str) -> pd.DataFrame:
    """timestamp x region matrix of one raw column."""
    return pd.DataFrame({region: frame[column] for region, frame in frames.items()})


def make_national_features(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Country-wide state as of the forecast instant, identical for all regions.

        national_active_lag1   regions under alert at instant t (0..N)
        national_onsets_lag1   alerts started anywhere during the last hour
        national_minutes_lag1  total alert minutes anywhere during the last hour
        national_onsets_roll3  alerts started anywhere over the last 3 hours

    Order of operations: shift each region's column along TIME, then sum
    ACROSS regions (axis=1). skipna=False keeps the warm-up row NaN.
    The count includes the region being forecast; its own state is also
    available separately as active_lag1.
    """
    active = _past(_wide(frames, "active_at_end"), 1)
    onsets = _past(_wide(frames, "n_onsets"), 1)
    minutes = _past(_wide(frames, "alert_minutes"), 1)

    N = pd.DataFrame(index=active.index)
    N["national_active_lag1"] = active.sum(axis=1, skipna=False)
    N["national_onsets_lag1"] = onsets.sum(axis=1, skipna=False)
    N["national_minutes_lag1"] = minutes.sum(axis=1, skipna=False)
    N["national_onsets_roll3"] = (
        N["national_onsets_lag1"].rolling(SPATIAL_ROLL, min_periods=SPATIAL_ROLL).sum()
    )
    return N


def make_neighbor_features(frames: dict[str, pd.DataFrame], region: str) -> pd.DataFrame:
    """Optional: state of the regions that border `region` (see ADJACENCY).

        neighbors_active_lag1   bordering regions under alert at instant t
        neighbors_active_share  the same as a share of the region's neighbours
        neighbors_onsets_roll3  alert starts in bordering regions, last 3 hours
    """
    nbrs = [r for r in ADJACENCY.get(region, ()) if r in frames]
    out = pd.DataFrame(index=frames[region].index)
    if not nbrs:
        return out.assign(neighbors_active_lag1=0.0, neighbors_active_share=0.0,
                          neighbors_onsets_roll3=0.0)
    sub = {r: frames[r] for r in nbrs}
    active = _past(_wide(sub, "active_at_end"), 1).sum(axis=1, skipna=False)
    onsets = _past(_wide(sub, "n_onsets"), 1).sum(axis=1, skipna=False)
    out["neighbors_active_lag1"] = active
    out["neighbors_active_share"] = active / len(nbrs)
    out["neighbors_onsets_roll3"] = onsets.rolling(SPATIAL_ROLL, min_periods=SPATIAL_ROLL).sum()
    return out


# --------------------------------------------------------------------------- #
# Panel assembly
# --------------------------------------------------------------------------- #
def make_features(frames: dict[str, pd.DataFrame], use_adjacency: bool = False) -> pd.DataFrame:
    """Causal feature matrix for the whole panel.

    Returns a DataFrame indexed by (timestamp, region), sorted by timestamp,
    with a categorical `region` column. Stacking is the LAST step: every shift
    and rolling window has already been applied on time-indexed data.
    """
    regions = sorted(frames)
    index = frames[regions[0]].index
    for region in regions:
        if not frames[region].index.equals(index):
            raise ValueError(f"{region}: not on the shared hourly grid")

    calendar = make_calendar_features(index)
    national = make_national_features(frames)

    blocks = {}
    for region in regions:
        block = make_own_features(frames[region])       # this region's frame only
        block = block.join(calendar).join(national)
        if use_adjacency:
            block = block.join(make_neighbor_features(frames, region))
        blocks[region] = block

    X = pd.concat(blocks, names=["region", "timestamp"])
    X = X.swaplevel().sort_index()                      # (timestamp, region)
    X["region"] = pd.Categorical(X.index.get_level_values("region"), categories=regions)
    return X


def build_panel(frames: dict[str, pd.DataFrame], horizon: int = 1,
                use_adjacency: bool = False, dropna: bool = True
                ) -> tuple[pd.DataFrame, pd.Series]:
    """Aligned (X, y) for the panel, indexed by (forecast instant, region).

    dropna removes warm-up rows (lags not yet filled) and, for horizon > 1,
    tail rows whose target window is incomplete. Not a fitted operation.
    """
    X = make_features(frames, use_adjacency=use_adjacency)
    y = pd.concat({r: make_target(f, horizon) for r, f in frames.items()},
                  names=["region", "timestamp"]).swaplevel().sort_index()
    assert X.index.equals(y.index), "features and target are misaligned"
    if dropna:
        keep = X.notna().all(axis=1) & y.notna()
        X, y = X[keep], y[keep].astype("int8")
    return X, y


# --------------------------------------------------------------------------- #
# Empirical leakage tests
# --------------------------------------------------------------------------- #
def _scramble(frame: pd.DataFrame, pos: int, rng: np.random.Generator) -> pd.DataFrame:
    """Copy of `frame` with every value at rows >= pos replaced by noise."""
    out = frame.copy()
    for col in frame.columns:
        noise = rng.integers(0, 61, size=len(frame) - pos).astype(frame[col].dtype)
        out.iloc[pos:, out.columns.get_loc(col)] = noise
    return out


def check_causality(frames: dict[str, pd.DataFrame], n_cuts: int = 10, seed: int = 0,
                    **kwargs) -> None:
    """Temporal leakage test. Raises AssertionError on any look-ahead.

    For random cut-off instants t0: overwrite every raw value at rows >= t0 with
    noise IN EVERY REGION, rebuild the panel features, and require all rows
    with timestamp <= t0 to be bit-identical. Rows at t0 are included on
    purpose: bin t0 is scrambled everywhere, so a feature at (t0, r) that read
    bin t0 of r or of any other region makes the comparison fail.
    """
    rng = np.random.default_rng(seed)
    reference = make_features(frames, **kwargs)
    index = next(iter(frames.values())).index
    warmup = max(max(SEASONAL_LAGS), max(ROLLING_WINDOWS)) + 1

    for pos in rng.integers(warmup, len(index) - 1, size=n_cuts):
        scrambled = {r: _scramble(f, pos, rng) for r, f in frames.items()}
        rebuilt = make_features(scrambled, **kwargs)
        past = reference.index.get_level_values("timestamp") <= index[pos]
        pd.testing.assert_frame_equal(reference[past], rebuilt[past],
                                      obj=f"features up to cut {index[pos]}")


def check_region_isolation(frames: dict[str, pd.DataFrame], n_regions: int = 3,
                           seed: int = 0, **kwargs) -> None:
    """Cross-region bleed test. Raises AssertionError if one region's data
    reaches another region's own-history features.

    Rewrite the ENTIRE history of one region with noise and rebuild. The
    own-history and calendar columns of every other region must be unchanged.
    National and neighbour aggregates are expected to change and are excluded:
    they are the one intended cross-region channel.
    """
    rng = np.random.default_rng(seed)
    reference = make_features(frames, **kwargs)
    private = [c for c in reference.columns if not c.startswith(SPATIAL_PREFIXES)]
    regions = sorted(frames)

    for victim in rng.choice(regions, size=min(n_regions, len(regions)), replace=False):
        altered = dict(frames)
        altered[victim] = _scramble(frames[victim], 0, rng)
        rebuilt = make_features(altered, **kwargs)
        others = reference.index.get_level_values("region") != victim
        pd.testing.assert_frame_equal(reference.loc[others, private], rebuilt.loc[others, private],
                                      obj=f"own-history features after rewriting {victim}")
        # The test must have teeth: the victim's own features do change.
        assert not reference.loc[~others, private].equals(rebuilt.loc[~others, private])


if __name__ == "__main__":
    import logging

    from data_loader import load_hourly_multi

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    frames = load_hourly_multi()
    check_causality(frames)
    check_region_isolation(frames)
    print("causality and region-isolation checks passed")
    X, y = build_panel(frames)
    print(X.shape, f"positive rate={y.mean():.4f}")
    print(X.tail(3).T.to_string())
