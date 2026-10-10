"""
data_loader.py
==============
Download the Vadimkin air-raid-siren dataset, filter one region, and turn the
raw alert intervals into a gap-free hourly time series.

Output contract (the only thing downstream modules rely on)
-----------------------------------------------------------
A DataFrame indexed by a tz-aware UTC DatetimeIndex with freq="h".
The row labelled `t` describes the *completed* hour bin [t, t+1h):

    alert_minutes  float  minutes of alert inside the bin (0..60)
    n_onsets       int    number of alerts that STARTED inside the bin
    alert_onset    int8   1 if n_onsets > 0 else 0
    active_at_end  int8   1 if an alert is running at instant t+1h

Every column of row `t` becomes known only at t+1h. features.py therefore
never reads row `t` when building the features of row `t` (see its docstring).

Leakage notes for this module
-----------------------------
* An interval is clipped to each bin it touches, so a bin's values depend only
  on what happened inside that bin. An alert's eventual end time or total
  duration never flows backwards into earlier bins.
* Nothing here is "fitted": no scaling, imputation or statistics. The module is
  a pure, deterministic transformation and is safe to run on the full history.
* The grid stops at the last *complete* hour before the dataset's as-of time,
  so a half-observed final hour is never mistaken for a quiet one.
"""

from __future__ import annotations

import logging
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_URL = (
    "https://raw.githubusercontent.com/Vadimkin/"
    "ukrainian-air-raid-sirens-dataset/main/datasets/"
)
FILES = {"official": "official_data_en.csv", "volunteer": "volunteer_data_en.csv"}

# Fail loudly if the upstream schema changes.
EXPECTED_COLUMNS = {
    "official": ["oblast", "raion", "hromada", "level", "started_at", "finished_at", "source"],
    "volunteer": ["region", "started_at", "finished_at", "naive"],
}

# Permanent alerts are NOT listed in the dataset, so these regions would look
# like "almost never alerted". Refuse to model them. (Crimea is absent entirely.)
PERMANENT_ALERT_REGIONS = {"Luhanska oblast"}

DEFAULT_REGION = "Kyiv City"


# --------------------------------------------------------------------------- #
# 1. Download / read
# --------------------------------------------------------------------------- #
def download_raw(source: str = "official", data_dir: str | Path = "data",
                 max_age_hours: float | None = 24.0) -> Path:
    """Fetch the CSV into `data_dir` (cached). Re-download if older than
    `max_age_hours`; pass None to always reuse an existing file."""
    if source not in FILES:
        raise ValueError(f"source must be one of {list(FILES)}, got {source!r}")
    path = Path(data_dir) / FILES[source]
    path.parent.mkdir(parents=True, exist_ok=True)

    fresh = path.exists() and (
        max_age_hours is None or (time.time() - path.stat().st_mtime) < max_age_hours * 3600
    )
    if not fresh:
        url = BASE_URL + FILES[source]
        logger.info("Downloading %s", url)
        tmp = path.with_suffix(".tmp")
        with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
            fh.write(resp.read())
        tmp.replace(path)  # atomic: never leave a half-written cache file
    return path


def read_events(path: str | Path, source: str = "official",
                naive_policy: str = "drop") -> tuple[pd.DataFrame, pd.Timestamp]:
    """Read a raw CSV into a tidy event table for ALL regions.

    Returns
    -------
    events : DataFrame[region, started_at, finished_at]  (UTC, de-duplicated)
    as_of  : latest `finished_at` in the whole file = the moment up to which the
             dataset is known to be complete. Taken across all regions so that a
             quiet spell in the target region is still recorded as zeros.

    naive_policy (volunteer source only)
        Rows with naive=True never received an "all clear" message; upstream
        fabricates finished_at = started_at + 30 min.
        "drop" (default): discard them. For Kyiv City only ~10% of them match an
                 official alert start, so most are not reliable events at all.
        "keep": retain them with the synthetic 30-minute end.
    """
    df = pd.read_csv(path)
    missing = set(EXPECTED_COLUMNS[source]) - set(df.columns)
    if missing:
        raise ValueError(
            f"{path}: schema changed, missing columns {sorted(missing)}; got {list(df.columns)}"
        )

    if source == "official":
        # Region = oblast: raion- and hromada-level alerts roll up to their
        # oblast ("any alert anywhere in the oblast"), which keeps one
        # consistent series across the 2025 switch to raion-level alerts.
        df = df.rename(columns={"oblast": "region"})
    else:
        if naive_policy not in ("drop", "keep"):
            raise ValueError("naive_policy must be 'drop' or 'keep'")
        naive = df["naive"].astype(str).str.lower().eq("true")
        logger.info("volunteer data: %d naive rows (%s)", int(naive.sum()), naive_policy)
        if naive_policy == "drop":
            df = df[~naive]

    for col in ("started_at", "finished_at"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    events = df[["region", "started_at", "finished_at"]]
    n0 = len(events)
    # Rows without an end are still-open (censored) alerts: we cannot place
    # them on the grid without inventing an end time, so drop them.
    events = events.dropna()
    events = events[events["finished_at"] > events["started_at"]]
    # The official file contains a large block of exact duplicate rows.
    events = events.drop_duplicates()
    logger.info("%s: %d raw rows -> %d clean events", source, n0, len(events))

    as_of = events["finished_at"].max()
    return events.sort_values("started_at").reset_index(drop=True), as_of


# --------------------------------------------------------------------------- #
# 2. Interval algebra
# --------------------------------------------------------------------------- #
def _secs(x) -> np.ndarray:
    """Timestamps -> integer seconds since epoch (the data has 1 s resolution)."""
    return pd.DatetimeIndex(x).as_unit("s").asi8


def merge_intervals(events: pd.DataFrame, gap: pd.Timedelta = pd.Timedelta(0)) -> pd.DataFrame:
    """Union overlapping intervals into disjoint ones.

    Needed for oblast-level aggregation: when raion A is already under alert
    and raion B joins, that is NOT a new onset for the oblast. After merging,
    every `started_at` is a genuine quiet -> alert transition.
    Intervals separated by <= `gap` are fused as well.
    """
    ev = events.sort_values("started_at")
    if len(ev) == 0:
        return ev[["started_at", "finished_at"]]
    start, end = _secs(ev["started_at"]), _secs(ev["finished_at"])

    running_end = np.maximum.accumulate(end)
    # A new block begins where an interval starts after everything before it ended.
    new_block = np.r_[True, start[1:] > running_end[:-1] + int(gap.total_seconds())]
    block_id = np.cumsum(new_block)
    merged = (
        pd.DataFrame({"started_at": start, "finished_at": end, "block": block_id})
        .groupby("block")
        .agg(started_at=("started_at", "min"), finished_at=("finished_at", "max"))
        .reset_index(drop=True)
    )
    for col in merged:
        merged[col] = pd.to_datetime(merged[col], unit="s", utc=True)
    return merged


def to_hourly(intervals: pd.DataFrame, start: pd.Timestamp, as_of: pd.Timestamp) -> pd.DataFrame:
    """Rasterise disjoint, sorted alert intervals onto a complete hourly grid.

    Grid: hours [t, t+1h) with  ceil(start) <= t  and  t+1h <= floor(as_of).
    Hours with no alert are explicit zeros (a missing row would silently break
    every positional shift() downstream).
    """
    first = start.ceil("h")
    last_edge = as_of.floor("h")
    edges = pd.date_range(first, last_edge, freq="h")  # bin edges, len = n_bins + 1
    if len(edges) < 2:
        raise ValueError("Not enough data to build a single hourly bin")
    index = edges[:-1].rename("timestamp")

    s, e, edge = _secs(intervals["started_at"]), _secs(intervals["finished_at"]), _secs(edges)
    dur = e - s
    cum_before = np.r_[0, np.cumsum(dur)]  # total alert seconds in intervals 0..i-1

    # A(x) = total alert seconds before instant x. Because intervals are
    # disjoint and sorted, only the last interval starting at/before x can be
    # partially covered.
    i = np.searchsorted(s, edge, side="right") - 1  # last interval with start <= edge
    has = i >= 0
    ic = np.clip(i, 0, None)
    covered = np.where(has, cum_before[ic] + np.minimum(edge - s[ic], dur[ic]), 0)
    alert_minutes = np.diff(covered) / 60.0

    # Alert running exactly at an edge  <=>  start <= edge < end.
    active_at_edge = has & (edge < e[ic])

    # Onsets per bin: an alert starting at instant x belongs to bin floor(x).
    onset_bin = pd.DatetimeIndex(intervals["started_at"]).floor("h")
    n_onsets = (
        pd.Series(1, index=onset_bin).groupby(level=0).sum().reindex(index, fill_value=0)
    )

    hourly = pd.DataFrame(
        {
            "alert_minutes": alert_minutes,
            "n_onsets": n_onsets.to_numpy(dtype="int64"),
            "alert_onset": (n_onsets.to_numpy() > 0).astype("int8"),
            "active_at_end": active_at_edge[1:].astype("int8"),  # state at t+1h
        },
        index=index,
    )
    return hourly


# --------------------------------------------------------------------------- #
# 3. Public entry points
# --------------------------------------------------------------------------- #
def load_hourly_multi(regions: list[str] | None = None, source: str = "official",
                      data_dir: str | Path = "data", naive_policy: str = "drop",
                      max_age_hours: float | None = 24.0,
                      merge_gap_minutes: float = 0.0) -> dict[str, pd.DataFrame]:
    """Download (if needed) and return {region: hourly frame} for several regions.

    regions=None loads every region in the dataset except those under a
    permanent, unlisted alert (PERMANENT_ALERT_REGIONS): the national panel.

    All frames share ONE index: the grid start and the as-of cut-off are taken
    from the whole dataset, not per region. Row `t` therefore means the same
    hour in every frame, which is what makes cross-region shift() features
    safe: a positional lag of k rows is a lag of k hours for every region.
    """
    path = download_raw(source, data_dir, max_age_hours)
    events, as_of = read_events(path, source, naive_policy)
    available = sorted(events["region"].unique())
    start = events["started_at"].min()

    if regions is None:
        regions = [r for r in available if r not in PERMANENT_ALERT_REGIONS]

    out: dict[str, pd.DataFrame] = {}
    for region in regions:
        if region in PERMANENT_ALERT_REGIONS:
            raise ValueError(f"{region!r} is under a permanent alert that the dataset does not list")
        if region not in available:
            raise ValueError(f"Unknown region {region!r}. Available: {available}")

        intervals = merge_intervals(
            events.loc[events["region"] == region, ["started_at", "finished_at"]],
            gap=pd.Timedelta(minutes=merge_gap_minutes),
        )
        hourly = to_hourly(intervals, start, as_of)

        # --- integrity checks: cheap, and they guard every later shift() -----
        assert hourly.index.is_monotonic_increasing and hourly.index.is_unique
        assert (hourly.index[1:] - hourly.index[:-1] == pd.Timedelta(hours=1)).all(), "gap in grid"
        assert hourly["alert_minutes"].between(0, 60 + 1e-9).all()
        assert not hourly.isna().any().any()

        hourly.attrs.update(region=region, source=source, as_of=str(as_of),
                            n_alerts=int(len(intervals)))
        logger.debug("%s/%s: %d alerts, onset rate %.3f",
                     region, source, len(intervals), hourly["alert_onset"].mean())
        out[region] = hourly

    first = next(iter(out.values())).index
    assert all(h.index.equals(first) for h in out.values()), "regions are not on one grid"
    logger.info("%s: %d regions x %d hourly rows, %s -> %s",
                source, len(out), len(first), first[0], first[-1])
    return out


def load_hourly(region: str = DEFAULT_REGION, **kwargs) -> pd.DataFrame:
    """Hourly series for a single region (see load_hourly_multi for options)."""
    return load_hourly_multi([region], **kwargs)[region]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    h = load_hourly()
    print(h.tail(8))
    print(h.describe().T[["mean", "min", "max"]])
    print("as_of:", h.attrs["as_of"])
