"""Analyst: summarizes what the evidence bought so far says about the labor market."""

from __future__ import annotations

from src.agents.common import RULES, clean_direction, task_header
from src.agents.evidence import evidence_block
from src.agents.llm import Usage, ask_json

SYSTEM = (
    "ROLE: analyst. You are the data analyst on an economic advisory panel. "
    "Summarize, without forecasting, what each piece of evidence says about where unemployment is heading. "
    + RULES
)
SCHEMA = (
    'Reply as JSON: {"summary": "<2-3 sentences>", '
    '"signals": [{"series_id": "<id>", "direction": "up|down|neutral"}]} '
    'where direction is the way that evidence points the unemployment rate.'
)


def summarize(client, episode, target_month, evidence: dict, usage: Usage, temperature=0.3, seed=None) -> dict:
    user = f"{task_header(episode, target_month)}\n\nEvidence:\n{evidence_block(evidence)}\n\n{SCHEMA}"
    out = ask_json(client, SYSTEM, user, usage, temperature, 300, seed, role="analyst")
    signals = [
        {"series_id": str(s.get("series_id", "")).upper(), "direction": clean_direction(s.get("direction"))}
        for s in out.get("signals", []) or [] if isinstance(s, dict)
    ]
    return {"summary": str(out.get("summary", ""))[:800], "signals": [s for s in signals if s["series_id"] in evidence]}
