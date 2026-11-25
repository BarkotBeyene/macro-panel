"""Shared prompt pieces and reply cleaning for all roles."""

from __future__ import annotations

import pandas as pd

PROMPT_VERSION = "panel-v1"

RULES = (
    "Rules: use only the evidence shown; data published after the forecast date are unknown to you. "
    "Unemployment moves slowly: month-to-month changes are usually 0.0 to 0.2 points. "
    "Reply with one JSON object and nothing else."
)


def task_header(episode, target_month) -> str:
    ep = pd.Timestamp(episode)
    tm = pd.Timestamp(str(target_month)[:7] + "-01")
    nxt = tm + pd.offsets.MonthBegin(1)
    return (
        f"Forecast date: {ep:%Y-%m-%d}, 5 p.m. ET.\n"
        f"Task: nowcast the U.S. unemployment rate (seasonally adjusted, %) for {tm:%B %Y}, "
        f"as BLS will first publish it in early {nxt:%B %Y}."
    )


def as_rate(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if 0.0 < v < 30.0 else None


def clean_direction(d) -> str:
    d = str(d or "").strip().lower()
    return d if d in ("up", "down", "neutral") else "neutral"


def clean_interval(point, lo, hi) -> tuple[float | None, float | None]:
    lo, hi = as_rate(lo), as_rate(hi)
    if point is None or lo is None or hi is None:
        return None, None
    lo, hi = min(lo, hi), max(lo, hi)
    return min(lo, point), max(hi, point)


def clean_cited(cited, evidence_keys) -> list[dict]:
    """Keep cited items that the agent actually saw; ranks 1..k in the order given."""
    out, seen = [], set()
    for c in cited or []:
        if not isinstance(c, dict):
            continue
        sid = str(c.get("series_id", "")).strip().upper()
        if sid in evidence_keys and sid not in seen:
            seen.add(sid)
            out.append({"series_id": sid, "direction": clean_direction(c.get("direction"))})
    try:
        out.sort(key=lambda c: next(float(x.get("rank", 99)) for x in cited
                                    if isinstance(x, dict) and str(x.get("series_id", "")).strip().upper() == c["series_id"]))
    except (TypeError, ValueError):
        pass
    for i, c in enumerate(out, 1):
        c["rank"] = i
    return out
