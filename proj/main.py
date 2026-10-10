"""
main.py
=======
Entry point: national panel, strictly causal features, walk-forward evaluation.

    python main.py                       # full pipeline, all valid regions
    python main.py --adjacency           # add bordering-region features
    python main.py --regions "Kyiv City" "Kyivska oblast"
    python main.py --embargo-hours 48 --n-splits 4

Steps
-----
1. Download / read the official dataset and build one hourly grid per region.
2. Run the two leakage tests. The run aborts if either fails.
3. Build the (timestamp, region) panel: X, y.
4. Walk-forward cross-validation of the baselines and LightGBM.
5. Print the comparison and write the detailed results to CSV.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from data_loader import load_hourly_multi
from evaluate import cross_validate, per_fold_table, per_region_table, summarise
from features import build_panel, check_causality, check_region_isolation
from model import PRIMARY_MODEL


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Air raid alert onset forecasting (hourly, per region).")
    p.add_argument("--source", default="official", choices=["official", "volunteer"])
    p.add_argument("--regions", nargs="*", default=None,
                   help="Subset of regions; default is every region with usable data.")
    p.add_argument("--horizon", type=int, default=1, help="Forecast horizon in hours.")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--test-days", type=int, default=90)
    p.add_argument("--embargo-hours", type=int, default=24)
    p.add_argument("--adjacency", action="store_true",
                   help="Add features built from each region's bordering regions.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--skip-checks", action="store_true",
                   help="Skip the leakage tests (not recommended).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 100)

    # 1. Data ----------------------------------------------------------------
    frames = load_hourly_multi(args.regions, source=args.source, data_dir=args.data_dir)

    # 2. Leakage tests: refuse to evaluate features that fail them -----------
    if not args.skip_checks:
        check_causality(frames, use_adjacency=args.adjacency)
        check_region_isolation(frames, use_adjacency=args.adjacency)
        print("Leakage tests passed: no look-ahead, no bleed between regions.")

    # 3. Panel ---------------------------------------------------------------
    X, y = build_panel(frames, horizon=args.horizon, use_adjacency=args.adjacency)
    print(f"Panel: {len(X):,} rows = {X['region'].nunique()} regions x "
          f"{X.index.get_level_values('timestamp').nunique():,} hours, "
          f"{X.shape[1]} features, positive rate {y.mean():.4f}")

    # 4. Walk-forward evaluation ---------------------------------------------
    results, by_region, importance = cross_validate(
        X, y, n_splits=args.n_splits, test_days=args.test_days,
        embargo_hours=args.embargo_hours, horizon=args.horizon)

    # 5. Report --------------------------------------------------------------
    print("\n=== Pooled PR-AUC per fold ===")
    print(per_fold_table(results, "pr_auc").round(4).to_string())

    print("\n=== Mean over folds (bss = Brier skill vs region base rate) ===")
    print(summarise(results).round(4).to_string())

    print(f"\n=== {PRIMARY_MODEL}: per region, mean over folds ===")
    regional = per_region_table(by_region, PRIMARY_MODEL)
    print(regional.round(4).to_string())

    print(f"\n=== {PRIMARY_MODEL}: gain share, final fold ===")
    print(importance[PRIMARY_MODEL].round(3).to_string())

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "cv_results.csv", index=False)
    by_region.to_csv(out / "cv_results_by_region.csv", index=False)
    regional.to_csv(out / "primary_model_by_region.csv")
    print(f"\nResults written to {out}/")


if __name__ == "__main__":
    main()
