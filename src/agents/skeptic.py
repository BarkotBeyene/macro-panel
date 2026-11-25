"""Skeptic: challenges the Economist's view and names evidence that could change it."""

from __future__ import annotations

from src.agents.common import RULES, as_rate, task_header
from src.agents.evidence import evidence_block
from src.agents.llm import Usage, ask_json

SYSTEM = (
    "ROLE: skeptic. You are the skeptic on an economic advisory panel. Challenge the leading view: "
    "point out weak or noisy evidence, overreaction, and what is missing. " + RULES
)
SCHEMA = (
    'Reply as JSON: {"critique": "<2-3 sentences>", "alternative_forecast": <number>, '
    '"missing_evidence": ["<id from the menu>"]}'
)


def challenge(client, episode, target_month, evidence: dict, economist: dict, menu: str, usage: Usage,
              temperature=0.3, seed=None) -> dict:
    user = (
        f"{task_header(episode, target_month)}\n\nEvidence:\n{evidence_block(evidence)}\n\n"
        f"Economist's forecast: {economist.get('forecast')} "
        f"(80% interval {economist.get('lo80')} to {economist.get('hi80')}). "
        f"Argument: {economist.get('argument', '')}\n\n"
        f"Evidence not yet bought:\n{menu}\n\n{SCHEMA}"
    )
    out = ask_json(client, SYSTEM, user, usage, temperature, 300, seed, role="skeptic")
    missing = [str(m).upper() for m in out.get("missing_evidence", []) or []]
    return {"critique": str(out.get("critique", ""))[:800],
            "alternative_forecast": as_rate(out.get("alternative_forecast")), "missing_evidence": missing}
