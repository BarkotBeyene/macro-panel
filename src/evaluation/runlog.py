"""Shared run-log format for every forecasting system (baselines, single agent, panel).

Each run appends one JSON object per forecast to a ``.jsonl`` file under
``results/forecasts/<run_id>/runs.jsonl``. Only the first five fields are
required; agents fill in the rest.

    episode         str    forecast date, "YYYY-MM-DD" (the 20th)
    target_month    str    month being nowcast, "YYYY-MM"
    system          str    e.g. "no_change", "ar1", "chronos2", "single_agent", "panel"
    forecast        float  point forecast of first-release UNRATE (percent)
    lo80, hi80      float  10th and 90th percentiles (80% interval); may be null
    cited           list   [{"series_id": "PAYEMS", "direction": "up"|"down"|"neutral", "rank": 1}]
    acquired        list   series_ids the system requested from the Gatekeeper
    credits         float  evidence budget spent
    tokens          int    LLM tokens used (0 for baselines)
    n_calls         int    LLM calls made (0 for baselines)
    runtime_s       float  wall-clock seconds
    evidence        dict   panel/single agent only: {series_id: what the Chair was shown};
                           needed by the faithfulness test to rerun the Chair's decision
    model, prompt_version, seed, vintage   provenance (free text / int)

"direction" means which way the Chair says that evidence pushed the forecast:
"up" = it raised the unemployment forecast, "down" = it lowered it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REQUIRED = ("episode", "target_month", "system", "forecast")
OPTIONAL_DEFAULTS = {
    "lo80": None,
    "hi80": None,
    "cited": [],
    "acquired": [],
    "credits": 0.0,
    "tokens": 0,
    "n_calls": 0,
    "runtime_s": None,
}


def read_runs(*paths) -> pd.DataFrame:
    """Load one or more runs.jsonl files (or folders containing one) into a DataFrame."""
    rows = []
    for p in paths:
        p = Path(p)
        files = sorted(p.rglob("*.jsonl")) if p.is_dir() else [p]
        for f in files:
            for line in f.read_text().splitlines():
                if line.strip():
                    rows.append(json.loads(line))
    return validate_runs(pd.DataFrame(rows))


def validate_runs(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"run log is missing required fields: {missing}")
    out = df.copy()
    for col, default in OPTIONAL_DEFAULTS.items():
        if col not in out.columns:
            out[col] = [default if not isinstance(default, list) else [] for _ in range(len(out))]
        elif isinstance(default, list):
            out[col] = out[col].apply(lambda v: v if isinstance(v, list) else [])
    out["episode"] = pd.to_datetime(out["episode"]).dt.normalize()
    out["target_month"] = pd.to_datetime(out["target_month"]).dt.to_period("M").dt.to_timestamp()
    out["forecast"] = pd.to_numeric(out["forecast"], errors="coerce")
    for col in ("lo80", "hi80", "credits", "runtime_s"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ("tokens", "n_calls"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    if out.duplicated(["system", "episode", *(["seed"] if "seed" in out else [])]).any():
        raise ValueError("duplicate (system, episode, seed) rows in run log")
    return out


def write_runs(records, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for rec in records:
            fh.write(json.dumps(rec, default=str) + "\n")
