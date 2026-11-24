"""Run the forecasting baselines on every episode and save them in the run-log format.

Systems (each sees only the UNRATE history the Gatekeeper shows on the
forecast date, and nowcasts the target month):

    no_change   last visible value; 80% interval from past h-month changes
    ar1         AR(1) on monthly changes, fitted on that vintage's history
    chronos2    Chronos-2 (amazon/chronos-2), zero-shot, 10/50/90% quantiles
    timesfm3    TimesFM-3 (google/timesfm-3.0-pytorch), zero-shot, 10/50/90% quantiles

``h`` is the number of months from the last visible month to the target month
(usually 1; 2 when a release was late, e.g. November 2025).

Output, one JSON line per forecast (see docs/run_log_format.md):

    results/forecasts/<run_id>/runs.jsonl
    results/forecasts/<run_id>/run.json      what was run, data snapshot hash

Usage:

    python -m src.baselines.run_baselines                     # no_change and ar1 (no downloads)
    python -m src.baselines.run_baselines no_change ar1 chronos2 timesfm3
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.build_episodes import EPISODES_PATH
from src.build_evidence import EVIDENCE_PATH, TARGET
from src.gatekeeper import REPO_ROOT, Gatekeeper, load_gatekeeper

RESULTS_DIR = REPO_ROOT / "results" / "forecasts"
QUANTILES = (0.1, 0.5, 0.9)  # lo80, point, hi80
SEED = 0

# Pandemic months are excluded when estimating AR(1) and the no-change interval:
# a 10-point jump in April 2020 would otherwise dominate both.
PANDEMIC = (pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-01"))
AR1_SAMPLE_START = pd.Timestamp("1990-01-01")
NO_CHANGE_WINDOW_MONTHS = 120

Forecaster = Callable[[pd.Series, int], tuple[float, float, float]]


def _outside_pandemic(index: pd.DatetimeIndex) -> np.ndarray:
    return ~((index >= PANDEMIC[0]) & (index <= PANDEMIC[1]))


# ---- statistical baselines --------------------------------------------------
def no_change(history: pd.Series, h: int) -> tuple[float, float, float]:
    """Last value; interval from the 10th/90th percentiles of past h-month changes."""
    last = float(history.iloc[-1])
    changes = (history - history.shift(h)).dropna()
    changes = changes[_outside_pandemic(changes.index)].iloc[-NO_CHANGE_WINDOW_MONTHS:]
    lo, hi = np.quantile(changes, [QUANTILES[0], QUANTILES[2]])
    return last, last + float(lo), last + float(hi)


def ar1(history: pd.Series, h: int, n_paths: int = 5000) -> tuple[float, float, float]:
    """AR(1) on monthly changes: d_t = c + phi * d_{t-1} + e_t, fitted by OLS."""
    d = history.diff().dropna()
    d = d[d.index >= AR1_SAMPLE_START]
    pairs = pd.DataFrame({"y": d, "x": d.shift(1)}).dropna()
    keep = _outside_pandemic(pairs.index) & _outside_pandemic(pairs.index - pd.DateOffset(months=1))
    pairs = pairs[keep]
    X = np.column_stack([np.ones(len(pairs)), pairs["x"].to_numpy()])
    (c, phi), *_ = np.linalg.lstsq(X, pairs["y"].to_numpy(), rcond=None)
    sigma = float(np.std(pairs["y"].to_numpy() - X @ np.array([c, phi]), ddof=2))

    rng = np.random.default_rng(SEED)
    level = np.full(n_paths, float(history.iloc[-1]))
    prev = np.full(n_paths, float(d.iloc[-1]))
    for _ in range(h):
        prev = c + phi * prev + rng.normal(0.0, sigma, n_paths)
        level = level + prev
    # Point forecast: the expected path (no noise).
    point, p = float(history.iloc[-1]), float(d.iloc[-1])
    for _ in range(h):
        p = c + phi * p
        point += p
    lo, hi = np.quantile(level, [QUANTILES[0], QUANTILES[2]])
    return point, float(lo), float(hi)


# ---- foundation models (loaded only when requested) -------------------------
def make_chronos2(model_id: str = "amazon/chronos-2") -> Forecaster:
    import torch
    from chronos import BaseChronosPipeline

    pipe = BaseChronosPipeline.from_pretrained(model_id, device_map="cpu")

    def forecast(history: pd.Series, h: int) -> tuple[float, float, float]:
        ctx = torch.tensor(history.to_numpy(dtype="float32"))
        q, _ = pipe.predict_quantiles([ctx], prediction_length=h, quantile_levels=list(QUANTILES))
        lo, mid, hi = q[0].reshape(-1, h, len(QUANTILES))[0, h - 1].tolist()
        return mid, lo, hi

    forecast.model_id = model_id
    return forecast


def make_timesfm3(model_id: str = "google/timesfm-3.0-pytorch") -> Forecaster:
    from timesfm3 import TimesFM3Forecaster

    model = TimesFM3Forecaster.from_pretrained(model_id, device="cpu")
    levels = [round(float(q), 2) for q in model.config.quantiles]
    idx = [levels.index(q) for q in QUANTILES]

    def forecast(history: pd.Series, h: int) -> tuple[float, float, float]:
        out = model.predict(history.to_numpy(dtype="float32"), horizon=h, return_quantiles=True)
        qs = np.asarray(out.quantiles)
        qs = qs.reshape(-1, qs.shape[-1])[h - 1]
        off = qs.shape[-1] - len(levels)  # 1 if a mean column comes first
        lo, mid, hi = (float(qs[i + off]) for i in idx)
        return mid, lo, hi

    forecast.model_id = model_id
    return forecast


SYSTEMS: dict[str, Callable[[], Forecaster]] = {
    "no_change": lambda: no_change,
    "ar1": lambda: ar1,
    "chronos2": make_chronos2,
    "timesfm3": make_timesfm3,
}
MODEL_NAMES = {"no_change": "no-change", "ar1": "AR(1) on monthly changes, OLS"}


# ---- runner -----------------------------------------------------------------
def history_as_of(gk: Gatekeeper, cutoff) -> pd.Series:
    hist = gk.as_of(TARGET, cutoff)
    return pd.Series(hist["value"].to_numpy(dtype=float), index=pd.DatetimeIndex(hist["date"]))


def run(gk: Gatekeeper, episodes: pd.DataFrame, systems: dict[str, Forecaster]) -> list[dict]:
    records = []
    for e in episodes.itertuples():
        hist = history_as_of(gk, e.episode)
        target = pd.Timestamp(e.target_month + "-01")
        last = hist.index[-1]
        h = (target.year - last.year) * 12 + target.month - last.month
        for name, fn in systems.items():
            t0 = time.perf_counter()
            point, lo, hi = fn(hist, h)
            records.append({
                "episode": str(pd.Timestamp(e.episode).date()),
                "target_month": e.target_month,
                "system": name,
                "forecast": round(float(point), 4),
                "lo80": round(float(lo), 4),
                "hi80": round(float(hi), 4),
                "cited": [],
                "acquired": [TARGET],
                "credits": 0,
                "tokens": 0,
                "n_calls": 0,
                "runtime_s": round(time.perf_counter() - t0, 4),
                "model": getattr(fn, "model_id", MODEL_NAMES.get(name, name)),
                "prompt_version": None,
                "seed": SEED,
                "vintage": str(pd.Timestamp(e.episode).date()),
                "horizon_months": h,
                "last_known_date": f"{last:%Y-%m}",
            })
    return records


def main(names=None) -> Path:
    names = list(names or ["no_change", "ar1"])
    unknown = sorted(set(names) - set(SYSTEMS))
    if unknown:
        raise SystemExit(f"unknown systems {unknown}; choose from {sorted(SYSTEMS)}")
    systems = {n: SYSTEMS[n]() for n in names}
    gk = load_gatekeeper(EVIDENCE_PATH, series=[TARGET])
    episodes = pd.read_csv(EPISODES_PATH)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-baselines"
    out_dir = RESULTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    records = run(gk, episodes, systems)
    with (out_dir / "runs.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    meta = {
        "run_id": run_id,
        "systems": {n: getattr(f, "model_id", MODEL_NAMES.get(n, n)) for n, f in systems.items()},
        "episodes": len(episodes),
        "evidence_table_sha256": hashlib.sha256(Path(EVIDENCE_PATH).read_bytes()).hexdigest(),
        "episodes_sha256": hashlib.sha256(Path(EPISODES_PATH).read_bytes()).hexdigest(),
        "quantiles": QUANTILES,
        "pandemic_excluded": [str(PANDEMIC[0].date()), str(PANDEMIC[1].date())],
        "python": platform.python_version(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2))

    df = pd.DataFrame(records).merge(
        episodes[["target_month", "target_first_release"]], on="target_month")
    err = (df["forecast"] - df["target_first_release"]).abs()
    inside = (df["target_first_release"] >= df["lo80"]) & (df["target_first_release"] <= df["hi80"])
    summary = df.assign(abs_err=err, inside=inside).groupby("system").agg(
        n=("abs_err", "size"), mae=("abs_err", "mean"), coverage80=("inside", "mean"))
    print(summary.round(3).to_string())
    print(f"saved {len(records)} forecasts -> {out_dir.relative_to(REPO_ROOT)}")
    return out_dir


if __name__ == "__main__":
    main(sys.argv[1:])
