# Air Raid Alert Onset Forecasting for Ukraine

A leakage-free time-series pipeline that estimates, for every region of
Ukraine and every hour, the probability that a **new air raid alert starts
within the next hour**.

The project was built in two stages: a single-region MVP (Kyiv City) used to
prove that the feature and validation code cannot see the future, then a
national panel model covering 24 regions with one shared LightGBM classifier.

> This is an analytical exercise on public data. It is not a warning system
> and must never replace official alerts.

## Results

Walk-forward evaluation, 5 folds of 90 days each (14 Jun 2025 to 7 Sep 2026),
24-hour embargo, 24 regions, 938,592 region-hours, 8.3% positive rate.
Figures are means over the five folds.

| Model | PR-AUC (pooled) | PR-AUC (macro) | Brier | Brier skill |
|---|---|---|---|---|
| Global base rate | 0.090 | 0.092 | 0.0820 | -0.034 |
| Region base rate | 0.152 | 0.092 | 0.0792 | 0.000 |
| Persistence, raw 0/1 | 0.085 | 0.085 | 0.3309 | -3.237 |
| Persistence, fitted per region | 0.282 | 0.148 | 0.0761 | 0.041 |
| Hour-of-week climatology | 0.161 | 0.108 | 0.0793 | 0.000 |
| LightGBM, own history only | 0.319 | 0.190 | 0.0710 | 0.101 |
| **LightGBM global** (default) | **0.324** | **0.248** | **0.0699** | **0.116** |
| LightGBM global + `--adjacency` | 0.337 | 0.263 | 0.0691 | 0.126 |

How to read the columns:

- **PR-AUC (pooled)** is average precision over all region-hours. It is the
  primary metric, but it rewards a model for merely knowing which regions are
  alerted most often: the region base rate alone scores 0.152.
- **PR-AUC (macro)** is average precision inside each region, averaged over
  regions. It measures timing skill only. Its no-skill level is 0.092.
- **Brier skill** is relative to the region base rate of the same fold.
- ROC-AUC is deliberately not reported. With under 10% positives it is
  dominated by easy negatives.

What the numbers say:

- The global model beats every baseline on every metric and in every fold on
  macro PR-AUC and Brier score. It beats the region base rate on Brier score
  in 104 of 120 region-folds.
- Country-wide state matters mostly for timing: adding the national features
  lifts macro PR-AUC from 0.190 to 0.248, while pooled PR-AUC moves only from
  0.319 to 0.324 and is slightly lower in two of five folds.
- Bordering-region features (`--adjacency`) improve pooled PR-AUC in all five
  folds. They are opt-in because the specified design uses national
  aggregates only.

Per-region results are written to `results/primary_model_by_region.csv`.

## Quick start

```bash
pip install -r requirements.txt
python main.py                 # download data, run leakage tests, evaluate
python main.py --adjacency     # add bordering-region features
python main.py --regions "Kyiv City" "Kyivska oblast" --n-splits 4
python features.py             # leakage tests only
python eda_visuals.py          # three EDA charts, written to figures/
```

The full run takes about three minutes on two cores. The dataset is downloaded
once into `data/` and refreshed when older than 24 hours.

Tested with Python 3.13, pandas 3.0, numpy 2.5, scikit-learn 1.9 and
LightGBM 4.7.

## Repository layout

| File | Responsibility | Learns from data? |
|---|---|---|
| `data_loader.py` | Download, clean, merge alert intervals, build one hourly grid per region | No |
| `features.py` | Causal features, panel assembly, the two leakage tests | No |
| `model.py` | Walk-forward splitter, baselines, LightGBM | Yes, per training fold |
| `evaluate.py` | Metrics and cross-validation loop | No |
| `main.py` | Command-line entry point | No |
| `eda_visuals.py` | Exploratory charts: timing, duration, threat migration | No |

Keeping everything that is fitted in one module is intentional: there is only
one place where train/test contamination could occur.

## Problem formulation

- **Unit:** one row per (hour `t`, region).
- **Target:** `y = 1` if an alert *starts* in `[t, t+1h)`.
- **Why onset and not "alert active":** activity is dominated by persistence.
  If an alert is running at `t`, "active next hour" is nearly always true, so
  a model can score well without any skill.
- **Spatial grain:** oblast level, "any alert active anywhere in the oblast".
  Raion- and hromada-level alerts roll up to their oblast.

## Data

Source: [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset),
official file, all times UTC.

Cleaning decisions:

- **Duplicates.** The official file contains about 114,000 exact duplicate
  rows; they are dropped.
- **Overlapping alerts.** Intervals are merged per oblast, so a second raion
  joining an ongoing alert is not counted as a new oblast-level onset.
- **Permanent alerts.** Luhanska oblast is excluded and Crimea is absent:
  their permanent alerts are not listed in the data.
- **Volunteer file.** Not used. It disagrees with the official log on roughly
  a fifth of Kyiv City alert starts, and its `naive=True` rows carry a
  fabricated 30-minute end. Label quality was preferred over recency.
- **Complete grid.** Hours without alerts are explicit zeros, and the grid
  stops at the last complete hour before the dataset's latest timestamp.

## How leakage is prevented

### 1. Causal features

A model row at `t` is a forecast issued at instant `t`. The hourly row `t`
describes the bin `[t, t+1h)` and is only known at `t+1h`, so it is the future.

- Every history value enters through one function, `_past(data, lag)`, which
  calls `shift(lag)` and raises if `lag < 1`.
- Rolling windows are applied after the shift, so a 24-hour window at `t`
  covers bins `t-24 … t-1`.
- Alert intervals are clipped to the hours they touch, so an alert's eventual
  end time never reaches an earlier hour.
- Calendar features describe the target hour and are deterministic.

### 2. No bleed between regions

The usual panel bug is `df.shift(1)` on a stacked table, which hands the last
row of one region to the first row of the next. Here that cannot happen:

- Per-region features are computed on each region's own time-indexed frame,
  one region at a time. Regions are stacked only after all shifting is done.
- `_past()` refuses stacked (multi-index) input.
- Cross-region information is allowed in one place only: national and
  neighbour aggregates, built by shifting a timestamp × region matrix along
  time and then summing across regions.

### 3. Tests that fail the run

`main.py` runs both tests before any model is trained and aborts on failure.

- **`check_causality`** overwrites every raw value from a random cut-off
  onward with noise, in every region, rebuilds the features and requires all
  rows at or before the cut-off to be identical.
- **`check_region_isolation`** rewrites the entire history of one region and
  requires the own-history features of all other regions to be identical.

Both tests were validated against planted bugs: an unshifted national sum is
caught by the first, a stacked `shift(1)` by the second. As an end-to-end
control in the single-region study, feeding the current hour's alert minutes
as a feature sent PR-AUC from 0.10 to 0.99, so a real leak would be obvious.

### 4. Walk-forward validation with an embargo

```
|------------- train -------------|~24h~|--- test 1 ---|
|------------------- train -------------------|~24h~|--- test 2 ---|
```

- Splits are made on time, with one cut shared by all regions. A per-region
  or random split would leak through the national features.
- Training rows are those more than 24 hours before the test block. The
  embargo must be at least the horizon; 24 hours also prevents one multi-hour
  attack from sitting on both sides of a boundary.
- Every fold asserts the gap and zero overlap.

### 5. Fitting inside the fold

Base rates, the climatology table, LightGBM's handling of the `region`
category and the trees are all estimated in `fit()` on the training fold.
Models are re-created for every fold. Hyper-parameters are fixed constants and
were not tuned against the test folds.

## Features

| Block | Features |
|---|---|
| Own history | `active_lag1`, `minutes_lag1`, `onset_lag24`, `onset_lag168`, `onsets_roll24`, `onsets_roll168` |
| Calendar (Kyiv local time) | `hour_sin`, `hour_cos`, `dow` |
| Region | `region` as a native LightGBM categorical |
| National | `national_active_lag1`, `national_onsets_lag1`, `national_minutes_lag1`, `national_onsets_roll3` |
| Neighbours (`--adjacency`) | `neighbors_active_lag1`, `neighbors_active_share`, `neighbors_onsets_roll3` |

In the final fold the largest contributors by gain are `onsets_roll168` (42%),
`region` (12%), `active_lag1` (8%) and `national_active_lag1` (8%).

## Model

LightGBM binary classifier with small trees (depth 6, 31 leaves, at least 500
region-hours per leaf, 400 trees, learning rate 0.05).

Class imbalance is handled without reweighting. In the single-region study,
`scale_pos_weight` ranked slightly worse in every fold and inflated the average
prediction to 49% against a true rate near 6%. The weighted variant is still
available, with an analytic correction `p / (p + w(1 - p))` that restores
calibration without fitting anything extra.

## From single-region MVP to national panel

| Stage | Scope | PR-AUC | No-skill level |
|---|---|---|---|
| 1 | Kyiv City, own history | 0.096 | 0.065 |
| 2 | Kyiv City + 4 hand-picked neighbouring oblasts | 0.157 | 0.065 |
| 3 | 24 regions, one global model, national features | 0.324 pooled, 0.248 macro | 0.092 macro |

Stage 1 fixed the alignment convention and the causality test on a problem
small enough to verify by hand. Stage 2 showed that the signal is spatial:
with Kyiv quiet, the chance of an onset in the next hour rose from 3.6% with
no neighbour under alert to 25% with all four. Stage 3 replaced the hand-picked
neighbours with region-agnostic aggregates and added the region-isolation
test. Kyiv City inside the global model scores 0.152 (0.175 with
`--adjacency`), in line with its dedicated model.

## Limitations

**Probabilities are miscalibrated by region.** Averaged over the country the
model is close to calibrated (predicted over observed between 0.95 and 1.09 per
fold), but that hides offsetting errors:

- It **over-forecasts in central and western regions**: 13 to 15 of 24 regions
  are more than 10% too high in folds 2 to 5, and several western oblasts
  reach two to three times the observed rate in the latest fold.
- It **under-forecasts in frontline regions** such as Sumska, Chernihivska and
  Zaporizka, by 20 to 40%.

The likely cause is the move to raion-level alerts in late 2025 (documented
upstream as December 2025; in this data Kyivska oblast's last oblast-wide
alert is 4 November 2025). Under "any alert anywhere in the oblast", frontline
oblasts are now under alert most of the time, which raises the national
activity features, while onset rates in western regions fell. The same
national state therefore implies a lower risk in the rear than it did in the
training history. This attribution is consistent with the data but not
proven; attack patterns also changed over the period. Rankings (PR-AUC) are
unaffected; the probabilities should be recalibrated before being read as
literal risk.

**The onset target degenerates in near-permanent-alert regions.** At oblast
level Dnipropetrovska oblast was under alert more than 99% of the time in the
test period, and Kharkivska oblast 100% of the time from June to October 2025.
New onsets there are rare or absent, and their per-region metrics are not
meaningful. They are kept because the scope is the whole country; exclude them
with `--regions`, or model them at raion level.

**Other caveats**

- The official file's latest event is 7 September 2026; results say nothing
  about later weeks.
- Hourly resolution: a forecast is issued on the hour, and a threat can cross
  a neighbouring oblast and reach the target inside one bin.
- Five folds from one 15-month period; no confidence intervals. In the
  single-region study a model trained on shuffled labels varied by about
  ±0.008 PR-AUC between seeds.
- Hyper-parameters were chosen once and not tuned.

## Next steps

1. Recalibrate per region on the tail of each training fold, or weight recent
   data more heavily.
2. Model at raion level for the period after the switch.
3. Sub-hourly bins and longer horizons (`--horizon` is already supported).
4. Alert-type and threat-direction features, if a reliable source is added.
