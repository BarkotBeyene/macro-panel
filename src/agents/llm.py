"""LLM access for the panel: one chat client, JSON replies, usage counting.

Backends (same as the memorization probe):

    openai   any OpenAI-compatible server: Ollama, vLLM, LM Studio
             (default http://localhost:11434/v1, i.e. Ollama on the same machine)
    hf       local transformers model (src.evaluation.memorization.HFLocal)
    fake     deterministic stand-in for tests and dry runs (no model needed)

Settings come from arguments or, when omitted, from environment variables
MACRO_LLM_BACKEND, MACRO_LLM_MODEL, MACRO_LLM_URL. ``redecide`` in
``chair.py`` uses the same environment variables, so the faithfulness test
talks to the same model the panel used.
"""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.request
from dataclasses import dataclass, field

DEFAULT_BACKEND = "openai"
DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_URL = "http://localhost:11434/v1"


class OpenAIChat:
    """Minimal /v1/chat/completions client with optional JSON mode and seed."""

    def __init__(self, model: str, base_url: str, api_key: str | None = None, timeout: int = 600):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key, self.timeout = api_key or os.getenv("OPENAI_API_KEY", "none"), timeout

    def __call__(self, system, user, temperature, max_tokens, json_mode=True, seed=None):
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if seed is not None:
            body["seed"] = int(seed)
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        for attempt in range(2):  # first call may time out while the model loads
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    out = json.loads(resp.read())
                break
            except TimeoutError:
                if attempt == 1:
                    raise
        text = out["choices"][0]["message"]["content"]
        return text, int(out.get("usage", {}).get("total_tokens", 0))


class HFChat:
    """Wraps the memorization probe's local transformers client."""

    def __init__(self, model: str, load_4bit: bool = False):
        from src.evaluation.memorization import HFLocal

        self._inner = HFLocal(model, load_4bit=load_4bit)
        self._lock = threading.Lock()  # one GPU, one generate() at a time

    def __call__(self, system, user, temperature, max_tokens, json_mode=True, seed=None):
        if seed is not None:
            import torch

            torch.manual_seed(int(seed))
        with self._lock:
            return self._inner(system, user, temperature, max_tokens)


class FakeChat:
    """Deterministic replies keyed on the prompt's role tag. For tests and dry runs.

    It reads the evidence the prompt contains, so removing evidence changes its
    answer (lets the faithfulness pipeline be checked end to end without a GPU).
    """

    def __call__(self, system, user, temperature, max_tokens, json_mode=True, seed=None):
        role = re.search(r"ROLE:\s*(\w+)", system)
        role = role.group(1) if role else "unknown"
        last = _fake_last_rate(user)
        payems = "[PAYEMS]" in user
        icsa = "[ICSA]" in user
        nudge = (-0.05 if payems else 0.0) + (0.05 if icsa else 0.0)
        if role == "chairstep":
            menu = re.findall(r"^- (\w+) \(cost", user, flags=re.M)
            want = [s for s in ("PAYEMS", "ICSA", "CHRONOS2") if s in menu]
            reply = {"action": "request" if want else "commit", "requests": want, "reason": "labor data"}
        elif role == "analyst":
            reply = {"summary": "Labor data mixed.", "signals": [{"series_id": "PAYEMS", "direction": "down"}]}
        elif role == "economist":
            reply = {"forecast": round(last + nudge, 2), "lo80": round(last - 0.2, 2), "hi80": round(last + 0.2, 2),
                     "drivers": [{"series_id": "PAYEMS", "direction": "down"}] if payems else [],
                     "argument": "Payrolls are firm." if payems else "Little evidence."}
        elif role == "skeptic":
            reply = {"critique": "Claims may be noisy.", "alternative_forecast": round(last + 0.05, 2), "requests": []}
        elif role == "chair":
            cited = []
            if payems:
                cited.append({"series_id": "PAYEMS", "direction": "down", "rank": 1})
            if icsa:
                cited.append({"series_id": "ICSA", "direction": "up", "rank": len(cited) + 1})
            reply = {"forecast": round(last + nudge, 2), "lo80": round(last - 0.2, 2), "hi80": round(last + 0.2, 2),
                     "cited": cited, "explanation": "Weighed payrolls against claims."}
        else:
            reply = {}
        text = json.dumps(reply)
        return text, len(system.split()) + len(user.split()) + len(text.split())


def _fake_last_rate(user: str) -> float:
    m = re.search(r"latest published value: ([\d.]+)", user)
    return float(m.group(1)) if m else 4.0


def make_client(backend=None, model=None, url=None, load_4bit=False):
    backend = backend or os.getenv("MACRO_LLM_BACKEND", DEFAULT_BACKEND)
    model = model or os.getenv("MACRO_LLM_MODEL", DEFAULT_MODEL)
    url = url or os.getenv("MACRO_LLM_URL", DEFAULT_URL)
    if backend == "openai":
        return OpenAIChat(model, url)
    if backend == "hf":
        return HFChat(model, load_4bit=load_4bit)
    if backend == "fake":
        return FakeChat()
    raise ValueError(f"unknown backend {backend!r} (openai, hf, fake)")


# ---- JSON replies ------------------------------------------------------------
_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def parse_json(text: str) -> dict | None:
    """First JSON object in ``text`` (tolerates code fences and chatter)."""
    if not text:
        return None
    for candidate in (text, *(m.group(0) for m in _JSON_BLOCK.finditer(text))):
        try:
            out = json.loads(candidate.strip().strip("`"))
            if isinstance(out, dict):
                return out
        except json.JSONDecodeError:
            continue
    return None


@dataclass
class Usage:
    tokens: int = 0
    calls: int = 0
    parse_failures: int = 0
    transcript: list = field(default_factory=list)


def ask_json(client, system, user, usage: Usage, temperature=0.3, max_tokens=400, seed=None, role="") -> dict:
    """Call the model and return its JSON reply; one retry if it is not valid JSON."""
    for attempt in range(2):
        prompt = user if attempt == 0 else user + "\n\nYour previous reply was not valid JSON. Reply with one JSON object only."
        text, tokens = client(system, prompt, temperature, max_tokens, json_mode=True, seed=seed)
        usage.tokens += tokens
        usage.calls += 1
        out = parse_json(text)
        usage.transcript.append({"role": role, "reply": text[:2000]})
        if out is not None:
            return out
    usage.parse_failures += 1
    return {}
