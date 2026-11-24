"""Forecast accuracy and interval reliability (H3 and the secondary uncertainty outcome).

Scores every system in a run log against the first-release unemployment rate:

    MAE, RMSE                 point accuracy (percentage points)
    coverage80                share of actuals inside the 80% interval (target 0.80)
    width80                   mean interval width
    interval_score80          Gneiting and Raftery interval score (lower is better)
    directional_accuracy      did it call the change from last month's rate correctly

``compare_to`` adds a paired bootstrap 90% CI for each system's MAE minus a
reference system's MAE on the same episodes (negative = better than reference).

Usage:
    python -m src.evaluation.forecasting results/forecasts/<run_id> [more runs...] \
        --truth data/processed/forecast_episodes/episodes.csv --reference no_change
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.runlog import read_runs

ALPHA = 0.20  # 80% interval


def load_truth(path) -> pd.DataFrame:
    """Episodes file with target_month and target_first_release (plus optional last_known)."""
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    rename = {"target_first_release": "actual", "date": "episode"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "actual" not in df.columns:
        raise ValueError("truth file needs a target_first_release (or actual) column")
    df["target_month"] = pd.to_datetime(df["target_month"]).dt.to_period("M").dt.to_timestamp()
    keep = ["target_month", "actual"] + (["last_known"] if "last_known" in df.columns else [])
    return df[keep].drop_duplicates("target_month")


def interval_score(y, lo, hi, alpha=ALPHA):
    y, lo, hi = map(np.asarray, (y, lo, hi))
    return (hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)


def attach_truth(runs: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    df = runs.merge(truth, on="target_month", how="left")
    missing = df["actual"].isna().sum()
    if missing:
        print(f"note: {missing} forecasts have no first release (e.g. Oct 2025) and are not scored")
    return df.dropna(subset=["actual"])


def score(df: pd.DataFrame) -> pd.DataFrame:
    """One row per system. ``df`` must already have an ``actual`` column."""
    rows = []
    for system, g in df.groupby("system"):
        err = g["forecast"] - g["actual"]
        row = {
            "system": system,
            "n": len(g),
            "mae": err.abs().mean(),
            "rmse": np.sqrt((err**2).mean()),
            "bias": err.mean(),
        }
        iv = g.dropna(subset=["lo80", "hi80"])
        if len(iv):
            row["coverage80"] = ((iv["actual"] >= iv["lo80"]) & (iv["actual"] <= iv["hi80"])).mean()
            row["width80"] = (iv["hi80"] - iv["lo80"]).mean()
            row["interval_score80"] = interval_score(iv["actual"], iv["lo80"], iv["hi80"]).mean()
        if "last_known" in g.columns and g["last_known"].notna().any():
            k = g.dropna(subset=["last_known"])
            true_dir = np.sign((k["actual"] - k["last_known"]).round(2))
            pred_dir = np.sign((k["forecast"] - k["last_known"]).round(1))
            row["directional_accuracy"] = (true_dir == pred_dir).mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mae").reset_index(drop=True).round(4)


def compare_to(df: pd.DataFrame, reference: str, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Paired bootstrap of MAE(system) minus MAE(reference) on shared episodes."""
    rng = np.random.default_rng(seed)
    ref = df[df["system"] == reference].set_index("target_month")["forecast"]
    if ref.empty:
        raise ValueError(f"reference system {reference!r} not in run log")
    rows = []
    for system, g in df[df["system"] != reference].groupby("system"):
        g = g.groupby("target_month").agg(forecast=("forecast", "mean"), actual=("actual", "first"))
        shared = g.index.intersection(ref.index)
        if len(shared) < 3:
            continue
        diff = (g.loc[shared, "forecast"] - g.loc[shared, "actual"]).abs().to_numpy() - (
            ref.loc[shared] - g.loc[shared, "actual"]
        ).abs().to_numpy()
        boots = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
        rows.append(
            {
                "system": system,
                "vs": reference,
                "n_shared": len(shared),
                "mae_diff": diff.mean(),
                "ci90_low": np.quantile(boots, 0.05),
                "ci90_high": np.quantile(boots, 0.95),
                "share_better": (diff < 0).mean(),
            }
        )
    return pd.DataFrame(rows).round(4)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Score forecasts against first-release UNRATE")
    p.add_argument("runs", nargs="+", help="runs.jsonl files or run folders")
    p.add_argument("--truth", required=True, help="episodes file with target_month and target_first_release")
    p.add_argument("--reference", default="no_change")
    p.add_argument("--out", help="folder to write scores.csv and comparison.csv")
    args = p.parse_args(argv)

    df = attach_truth(read_runs(*args.runs), load_truth(args.truth))
    scores = score(df)
    print(scores.to_string(index=False))
    comp = pd.DataFrame()
    if args.reference in set(df["system"]):
        comp = compare_to(df, args.reference)
        print(f"\nMAE difference vs {args.reference} (negative = better), paired bootstrap 90% CI")
        print(comp.to_string(index=False))
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        scores.to_csv(out / "scores.csv", index=False)
        comp.to_csv(out / "comparison.csv", index=False)


if __name__ == "__main__":
    main()
