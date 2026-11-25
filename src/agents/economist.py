"""Economist: interprets the evidence and proposes a forecast with its drivers."""

from __future__ import annotations

from src.agents.common import RULES, as_rate, clean_direction, clean_interval, task_header
from src.agents.evidence import evidence_block
from src.agents.llm import Usage, ask_json

SYSTEM = (
    "ROLE: economist. You are the macroeconomist on an economic advisory panel. "
    "Interpret the evidence and propose a forecast with an 80% interval. " + RULES
)
SCHEMA = (
    'Reply as JSON: {"forecast": <number>, "lo80": <number>, "hi80": <number>, '
    '"drivers": [{"series_id": "<id>", "direction": "up|down"}], "argument": "<2-3 sentences>"} '
    "where drivers are the evidence items that most shape your forecast and direction is the way "
    "each pushes it relative to the latest published rate."
)


def propose(client, episode, target_month, evidence: dict, analyst: dict, usage: Usage,
            temperature=0.3, seed=None) -> dict:
    user = (
        f"{task_header(episode, target_month)}\n\nEvidence:\n{evidence_block(evidence)}\n\n"
        f"Analyst summary: {analyst.get('summary', '')}\n\n{SCHEMA}"
    )
    out = ask_json(client, SYSTEM, user, usage, temperature, 350, seed, role="economist")
    point = as_rate(out.get("forecast"))
    lo, hi = clean_interval(point, out.get("lo80"), out.get("hi80"))
    drivers = [
        {"series_id": str(d.get("series_id", "")).upper(), "direction": clean_direction(d.get("direction"))}
        for d in out.get("drivers", []) or [] if isinstance(d, dict)
    ]
    return {"forecast": point, "lo80": lo, "hi80": hi, "argument": str(out.get("argument", ""))[:800],
            "drivers": [d for d in drivers if d["series_id"] in evidence]}
