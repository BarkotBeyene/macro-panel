"""What the agents can see: the evidence menu, its costs, and how each item is shown.

Every item is built from the Gatekeeper as of the forecast date, so nothing
published after the cutoff can reach an agent. All descriptions for an episode
are built once, before any LLM call (``episode_evidence``); the agents only
choose which ones to buy. That keeps the Gatekeeper on the main thread and
lets the faithfulness test remove or swap items without touching data.

Menu (1 credit each, budget 4 per forecast; UNRATE's own history is free):

    PAYEMS JTSJOL INDPRO CPIAUCSL FEDFUNDS   monthly releases
    ICSA CCSA                                weekly claims
    CHRONOS2 TIMESFM3                        zero-shot foundation-model nowcasts,
                                             read from the saved baseline runs
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.gatekeeper import REPO_ROOT, Gatekeeper

TARGET = "UNRATE"
BUDGET = 4
COSTS = {
    "PAYEMS": 1, "ICSA": 1, "CCSA": 1, "JTSJOL": 1, "INDPRO": 1, "CPIAUCSL": 1, "FEDFUNDS": 1,
    "CHRONOS2": 1, "TIMESFM3": 1,
}
LABELS = {
    "UNRATE": "Unemployment rate, %, seasonally adjusted (the target's own history)",
    "PAYEMS": "Nonfarm payroll employment (BLS), monthly change in thousands of jobs",
    "ICSA": "Initial unemployment insurance claims (DOL), weekly",
    "CCSA": "Continued unemployment insurance claims (DOL), weekly",
    "JTSJOL": "Job openings, JOLTS (BLS), thousands, monthly",
    "INDPRO": "Industrial production index (Federal Reserve), monthly",
    "CPIAUCSL": "Consumer price index (BLS), monthly, 12-month inflation",
    "FEDFUNDS": "Effective federal funds rate, %, monthly average",
    "CHRONOS2": "Chronos-2 time-series model: zero-shot nowcast from the unemployment history alone",
    "TIMESFM3": "TimesFM-3 time-series model: zero-shot nowcast from the unemployment history alone",
}
MODEL_ITEMS = {"CHRONOS2": "chronos2", "TIMESFM3": "timesfm3"}
BASELINE_DIR = REPO_ROOT / "results" / "forecasts"


def _monthly(hist: pd.DataFrame, n: int = 6) -> pd.Series:
    return pd.Series(hist["value"].to_numpy(float), index=pd.DatetimeIndex(hist["date"])).iloc[-(n + 12):]


def _fmt_month(ts) -> str:
    return f"{pd.Timestamp(ts):%Y-%m}"


def describe_unrate(gk: Gatekeeper, cutoff) -> str:
    s = _monthly(gk.as_of(TARGET, cutoff))
    last12 = s.iloc[-12:]
    lines = [f"{_fmt_month(d)}: {v:.1f}" for d, v in last12.items()]
    return (
        f"{LABELS[TARGET]}. Last 12 published months:\n  " + "; ".join(lines)
        + f"\n  latest published value: {last12.iloc[-1]:.1f} (for {_fmt_month(last12.index[-1])})"
    )


def describe_series(gk: Gatekeeper, series_id: str, cutoff) -> str:
    hist = gk.as_of(series_id, cutoff)
    if hist.empty:
        return f"{LABELS[series_id]}. No data published by the forecast date."
    s = _monthly(hist, n=8)
    if series_id in ("ICSA", "CCSA"):
        s = pd.Series(hist["value"].to_numpy(float), index=pd.DatetimeIndex(hist["date"]))
        last = s.iloc[-6:]
        avg_now = s.iloc[-4:].mean()
        avg_3m = s.iloc[-17:-13].mean() if len(s) >= 17 else float("nan")
        weeks = "; ".join(f"week ending {d:%Y-%m-%d}: {v:,.0f}" for d, v in last.items())
        return (f"{LABELS[series_id]}. Last 6 weeks: {weeks}. 4-week average {avg_now:,.0f} "
                f"vs {avg_3m:,.0f} three months earlier.")
    if series_id == "PAYEMS":
        ch = s.diff().dropna().iloc[-6:]
        body = "; ".join(f"{_fmt_month(d)}: {v:+,.0f}k" for d, v in ch.items())
        return f"{LABELS[series_id]}. Last 6 months: {body}. 3-month average {ch.iloc[-3:].mean():+,.0f}k."
    if series_id in ("INDPRO", "CPIAUCSL"):
        yoy = (s / s.shift(12) - 1).dropna() * 100
        lvl = s.iloc[-6:]
        body = "; ".join(f"{_fmt_month(d)}: {v:.1f}" for d, v in lvl.items())
        tail = f" 12-month change {yoy.iloc[-1]:+.1f}%." if len(yoy) else ""
        return f"{LABELS[series_id]}. Last 6 months (index level): {body}.{tail}"
    lvl = s.iloc[-6:]
    fmt = "{:,.0f}" if series_id == "JTSJOL" else "{:.2f}"
    body = "; ".join(f"{_fmt_month(d)}: {fmt.format(v)}" for d, v in lvl.items())
    return f"{LABELS[series_id]}. Last 6 months: {body}."


def load_model_nowcasts(folder: Path = BASELINE_DIR) -> pd.DataFrame:
    """Saved Chronos-2 / TimesFM-3 baseline forecasts (episode, system, forecast, lo80, hi80)."""
    from src.evaluation.runlog import read_runs

    files = sorted(Path(folder).glob("*-baselines/runs.jsonl"))
    if not files:
        return pd.DataFrame(columns=["episode", "system", "forecast", "lo80", "hi80"])
    df = read_runs(*files)
    df = df[df["system"].isin(MODEL_ITEMS.values())]
    return df.drop_duplicates(["episode", "system"], keep="last")


def describe_model(item: str, nowcasts: pd.DataFrame, episode, target_month: str) -> str | None:
    row = nowcasts[(nowcasts["system"] == MODEL_ITEMS[item]) & (nowcasts["episode"] == pd.Timestamp(episode))]
    if row.empty:
        return None
    r = row.iloc[-1]
    return (f"{LABELS[item]}. Nowcast for {target_month}: {r['forecast']:.2f} "
            f"(80% interval {r['lo80']:.2f} to {r['hi80']:.2f}).")


def episode_evidence(gk: Gatekeeper, episode, target_month: str, nowcasts: pd.DataFrame | None = None) -> dict:
    """All evidence texts available on this forecast date: {item: text}. UNRATE included."""
    out = {TARGET: describe_unrate(gk, episode)}
    for sid in COSTS:
        if sid in MODEL_ITEMS:
            if nowcasts is not None:
                text = describe_model(sid, nowcasts, episode, target_month)
                if text:
                    out[sid] = text
        elif sid in gk.available_series():
            out[sid] = describe_series(gk, sid, episode)
    return out


def menu_text(available: dict, acquired: list[str]) -> str:
    rows = [f"- {sid} (cost {COSTS[sid]}): {LABELS[sid]}" for sid in COSTS if sid in available and sid not in acquired]
    return "\n".join(rows) if rows else "(nothing left to request)"


def evidence_block(evidence: dict) -> str:
    return "\n".join(f"[{sid}] {text}" for sid, text in evidence.items())


def buy(requests, available: dict, acquired: list[str], credits_left: int) -> tuple[list[str], int]:
    """Valid, affordable, new requests in the order asked. Returns (bought, credits spent)."""
    bought, spent = [], 0
    for sid in requests or []:
        sid = str(sid).strip().upper()
        if sid in available and sid in COSTS and sid not in acquired and sid not in bought:
            if spent + COSTS[sid] <= credits_left:
                bought.append(sid)
                spent += COSTS[sid]
    return bought, spent
