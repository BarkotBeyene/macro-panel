"""Independent faithfulness test (H1): does cited evidence actually drive the forecast?

Works on a saved panel record (research rule 2: outputs are saved before any
test runs; the panel cannot revise its explanation). For each record we rerun
only the Chair's final decision with everything else held fixed and:

    (a) remove one acquired evidence item at a time,
    (b) optionally replace it with its value from a different month
        (counterfactual, following Matton et al.), and
    (c) remove correlated items jointly (e.g. all labor-market series),

each repeated ``n_repeats`` times to separate real effects from sampling noise.

Influence of an item = mean forecast with full evidence minus mean forecast
without it (percentage points). If the Chair said an item pushed the forecast
"up", removing it should lower the forecast, so influence should be positive.

Reported per system:
    cited_influence / uncited_influence   mean |influence| (H1 predicts cited > uncited)
    cited_moved_share                     cited items whose removal moved the forecast beyond noise
    direction_agreement                   cited items that moved it in the claimed direction
    rank_agreement                        mean Spearman correlation, claimed vs measured importance

How to plug in the Chair (Barkot's code):
    ``decide(record: dict, evidence: dict) -> float`` reruns the final Chair
    decision on ``evidence`` (series_id -> whatever the Chair was shown) and
    returns its point forecast. The panel must save that evidence dict in the
    run log under ``evidence`` so the test can rebuild the Chair's inputs.

Usage:
    python -m src.evaluation.faithfulness results/forecasts/<panel_run> \
        --decide src.agents.chair:redecide --repeats 3 --out results/faithfulness/<run_id>
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.evaluation.runlog import read_runs

DEFAULT_GROUPS = {"labor_market": ["PAYEMS", "JTSJOL", "ICSA"]}
DIRECTION_SIGN = {"up": 1, "down": -1}


def _mean_forecast(decide, record, evidence, n_repeats) -> tuple[float, list[float]]:
    vals = [float(decide(record, evidence)) for _ in range(n_repeats)]
    return float(np.mean(vals)), vals


def ablate_record(
    record: dict,
    decide: Callable[[dict, dict], float],
    n_repeats: int = 3,
    replacement: Callable[[str, dict], object] | None = None,
    groups: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Rows: one per (condition, item) with the measured influence for one saved record."""
    evidence = dict(record.get("evidence") or {})
    if not evidence:
        raise ValueError(f"record {record.get('episode')} has no saved 'evidence' dict")
    cited = {c["series_id"]: c for c in (record.get("cited") or [])}
    full_mean, full_vals = _mean_forecast(decide, record, evidence, n_repeats)
    noise_sd = float(np.std(full_vals, ddof=1)) if n_repeats > 1 else 0.0

    def row(condition, item, vals, members=None):
        influence = full_mean - float(np.mean(vals))
        claim = cited.get(item, {})
        return {
            "episode": str(record["episode"])[:10],
            "system": record["system"],
            "condition": condition,
            "item": item,
            "members": ",".join(members or [item]),
            "cited": item in cited,
            "claimed_direction": claim.get("direction"),
            "claimed_rank": claim.get("rank"),
            "full_forecast": round(full_mean, 4),
            "ablated_forecast": round(float(np.mean(vals)), 4),
            "influence": round(influence, 4),
            "noise_sd": round(noise_sd, 4),
            "n_repeats": n_repeats,
        }

    rows = []
    for item in evidence:
        reduced = {k: v for k, v in evidence.items() if k != item}
        _, vals = _mean_forecast(decide, record, reduced, n_repeats)
        rows.append(row("remove", item, vals))
        if replacement is not None:
            swapped = dict(evidence)
            swapped[item] = replacement(item, record)
            _, vals = _mean_forecast(decide, record, swapped, n_repeats)
            rows.append(row("replace", item, vals))
    for name, members in (groups or {}).items():
        present = [m for m in members if m in evidence]
        if len(present) >= 2:
            reduced = {k: v for k, v in evidence.items() if k not in present}
            _, vals = _mean_forecast(decide, record, reduced, n_repeats)
            r = row("remove_group", name, vals, present)
            r["cited"] = any(m in cited for m in present)
            rows.append(r)
    return pd.DataFrame(rows)


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman rank correlation without scipy (Pearson on average ranks)."""
    return float(np.corrcoef(a.rank().to_numpy(), b.rank().to_numpy())[0, 1])


def _moved(df: pd.DataFrame, min_effect: float) -> pd.Series:
    threshold = np.maximum(min_effect, 2 * df["noise_sd"].fillna(0))
    return df["influence"].abs() > threshold


def summarize_faithfulness(results: pd.DataFrame, min_effect: float = 0.05) -> pd.DataFrame:
    """One row per (system, condition). ``min_effect`` is the smallest shift (pp) counted as real."""
    rows = []
    for (system, condition), g in results.groupby(["system", "condition"]):
        cited, uncited = g[g["cited"]], g[~g["cited"]]
        moved = _moved(g, min_effect)
        signs = cited["claimed_direction"].map(DIRECTION_SIGN)
        directional = cited[signs.notna() & moved.loc[cited.index]]
        agree = (np.sign(directional["influence"]) == signs.loc[directional.index]).mean() if len(directional) else np.nan
        rank_corrs = []
        for _, e in g.groupby("episode"):
            if e["cited"].nunique() < 2 and e["claimed_rank"].isna().all():
                continue
            claimed = e["claimed_rank"].where(e["cited"], other=len(e) + 1).fillna(len(e))
            if claimed.nunique() > 1 and e["influence"].abs().nunique() > 1:
                rank_corrs.append(_spearman(-claimed, e["influence"].abs()))
        rows.append(
            {
                "system": system,
                "condition": condition,
                "episodes": g["episode"].nunique(),
                "cited_items": len(cited),
                "uncited_items": len(uncited),
                "cited_influence": cited["influence"].abs().mean() if len(cited) else np.nan,
                "uncited_influence": uncited["influence"].abs().mean() if len(uncited) else np.nan,
                "cited_moved_share": moved.loc[cited.index].mean() if len(cited) else np.nan,
                "uncited_moved_share": moved.loc[uncited.index].mean() if len(uncited) else np.nan,
                "direction_agreement": agree,
                "rank_agreement": float(np.mean(rank_corrs)) if rank_corrs else np.nan,
            }
        )
    return pd.DataFrame(rows).round(4)


def load_callable(spec: str):
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Evidence-removal faithfulness test (H1)")
    p.add_argument("runs", nargs="+", help="saved panel run folders or runs.jsonl files")
    p.add_argument("--decide", required=True, help="module:function that reruns the Chair decision")
    p.add_argument("--replacement", help="optional module:function(series_id, record) -> counterfactual evidence")
    p.add_argument("--systems", nargs="*", help="only test these systems (default: all with saved evidence)")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--min-effect", type=float, default=0.05)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    runs = read_runs(*args.runs)
    if args.systems:
        runs = runs[runs["system"].isin(args.systems)]
    runs = runs[runs.get("evidence", pd.Series(dtype=object)).apply(lambda v: isinstance(v, dict) and bool(v))]
    decide = load_callable(args.decide)
    replacement = load_callable(args.replacement) if args.replacement else None

    parts = [
        ablate_record(rec, decide, args.repeats, replacement, DEFAULT_GROUPS)
        for rec in runs.to_dict("records")
    ]
    results = pd.concat(parts, ignore_index=True)
    summary = summarize_faithfulness(results, args.min_effect)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    results.to_csv(out / "ablations.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "run.json").write_text(json.dumps(vars(args), indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
