"""Chair: asks for more evidence or commits to the final decision.

The final decision is one separate call (``commit``) whose only inputs are the
task, the evidence and the saved discussion. That makes it re-runnable on
modified evidence with everything else held fixed, which is what the
faithfulness test (src/evaluation/faithfulness.py) needs:

    python -m src.evaluation.faithfulness results/forecasts/<panel_run> \
        --decide src.agents.chair:redecide --systems panel single_agent --repeats 3 \
        --out results/faithfulness/<run_id>

``redecide`` talks to the model named by MACRO_LLM_BACKEND / MACRO_LLM_MODEL /
MACRO_LLM_URL (defaults: Ollama on localhost, llama3.1:8b), the same settings
the orchestrator uses. It sends no seed, so repeats show sampling noise.
"""

from __future__ import annotations

import threading

from src.agents.common import RULES, as_rate, clean_cited, clean_interval, task_header
from src.agents.evidence import evidence_block
from src.agents.llm import Usage, ask_json, make_client

CHAIR_TEMPERATURE = 0.3

STEP_SYSTEM = (
    "ROLE: chairstep. You chair an economic advisory panel with an evidence budget. "
    "Decide whether the panel needs more evidence before deciding, or should commit now. " + RULES
)
STEP_SCHEMA = (
    'Reply as JSON: {"action": "request" or "commit", "requests": ["<id from the menu>"], "reason": "<1 sentence>"}'
)

PANEL_COMMIT_SYSTEM = (
    "ROLE: chair. You chair an economic advisory panel (analyst, economist, skeptic). "
    "Weigh the evidence and the discussion and make the panel's final decision. " + RULES
)
SINGLE_COMMIT_SYSTEM = (
    "ROLE: chair. You are a macroeconomic forecaster working alone. "
    "Weigh the evidence and make your final decision. " + RULES
)
COMMIT_SCHEMA = (
    'Reply as JSON: {"forecast": <number>, "lo80": <number>, "hi80": <number>, '
    '"cited": [{"series_id": "<id>", "direction": "up|down", "rank": <1 = most important>}], '
    '"explanation": "<2-4 sentences>"} '
    "where cited lists ONLY the evidence items that actually drove your forecast, most important first, "
    "and direction is the way each pushed your forecast relative to the latest published rate."
)


def discussion_text(debate: dict | None) -> str:
    if not debate:
        return ""
    a, e, s = debate.get("analyst") or {}, debate.get("economist") or {}, debate.get("skeptic") or {}
    return (
        "Panel discussion (final round):\n"
        f"- Analyst: {a.get('summary', '')}\n"
        f"- Economist: forecast {e.get('forecast')} (80% interval {e.get('lo80')} to {e.get('hi80')}). "
        f"{e.get('argument', '')}\n"
        f"- Skeptic: {s.get('critique', '')} Alternative forecast: {s.get('alternative_forecast')}.\n"
    )


SINGLE_STEP_SYSTEM = (
    "ROLE: chairstep. You are a macroeconomic forecaster working alone with an evidence budget. "
    "Decide whether you need more evidence before deciding, or should commit now. " + RULES
)


def next_step(client, episode, target_month, evidence, debate, menu, credits_left, usage: Usage, seed=None,
              single=False) -> dict:
    system = SINGLE_STEP_SYSTEM if single else STEP_SYSTEM
    user = (
        f"{task_header(episode, target_month)}\n\nEvidence bought so far:\n{evidence_block(evidence)}\n\n"
        f"{discussion_text(debate)}\nCredits left: {credits_left}. Evidence you can still buy:\n{menu}\n\n{STEP_SCHEMA}"
    )
    out = ask_json(client, system, user, usage, CHAIR_TEMPERATURE, 150, seed, role="chair_step")
    action = str(out.get("action", "commit")).lower()
    return {"action": "request" if action == "request" else "commit",
            "requests": [str(r).upper() for r in out.get("requests", []) or []],
            "reason": str(out.get("reason", ""))[:300]}


def commit(client, episode, target_month, evidence: dict, debate: dict | None, usage: Usage,
           seed=None, temperature=CHAIR_TEMPERATURE) -> dict:
    """The final decision. Same prompt for the panel (with discussion) and the single agent (without)."""
    system = PANEL_COMMIT_SYSTEM if debate else SINGLE_COMMIT_SYSTEM
    user = (
        f"{task_header(episode, target_month)}\n\nEvidence:\n{evidence_block(evidence)}\n\n"
        f"{discussion_text(debate)}\n{COMMIT_SCHEMA}"
    )
    out = ask_json(client, system, user, usage, temperature, 400, seed, role="chair")
    point = as_rate(out.get("forecast"))
    lo, hi = clean_interval(point, out.get("lo80"), out.get("hi80"))
    return {"forecast": point, "lo80": lo, "hi80": hi,
            "cited": clean_cited(out.get("cited"), set(evidence)),
            "explanation": str(out.get("explanation", ""))[:1200]}


# ---- faithfulness hook ---------------------------------------------------------
_client = None
_lock = threading.Lock()


def _shared_client():
    global _client
    with _lock:
        if _client is None:
            _client = make_client()
        return _client


def _as_dict(v):
    return v if isinstance(v, dict) else None


def redecide(record: dict, evidence: dict) -> float:
    """Rerun the Chair's final decision for this saved record on ``evidence``; return the point forecast.

    Everything except the evidence is held fixed: task, saved discussion (panel)
    or none (single agent), prompt and temperature. If the model's reply has no
    usable number, the saved forecast is returned so a parse failure is not
    mistaken for an effect of the evidence.
    """
    debate = _as_dict(record.get("debate")) if record.get("system") == "panel" else None
    target = record.get("target_month")
    target = f"{target:%Y-%m}" if hasattr(target, "strftime") else str(target)[:7]
    out = commit(_shared_client(), record["episode"], target, dict(evidence), debate, Usage(), seed=None)
    return out["forecast"] if out["forecast"] is not None else float(record["forecast"])
