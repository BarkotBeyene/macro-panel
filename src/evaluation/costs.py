"""Cost summary per system: evidence credits, LLM calls, tokens and runtime.

Usage:
    python -m src.evaluation.costs results/forecasts/<run_id> [more runs...]
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.evaluation.runlog import read_runs


def summarize_costs(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for system, g in runs.groupby("system"):
        n_acq = g["acquired"].apply(len)
        rows.append(
            {
                "system": system,
                "forecasts": len(g),
                "credits_per_forecast": g["credits"].mean(),
                "evidence_items_per_forecast": n_acq.mean(),
                "llm_calls_per_forecast": g["n_calls"].mean(),
                "tokens_per_forecast": g["tokens"].mean(),
                "tokens_total": int(g["tokens"].sum()),
                "runtime_s_per_forecast": g["runtime_s"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("tokens_per_forecast").reset_index(drop=True).round(2)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", help="CSV path for the summary")
    args = p.parse_args(argv)
    summary = summarize_costs(read_runs(*args.runs))
    print(summary.to_string(index=False))
    if args.out:
        summary.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
