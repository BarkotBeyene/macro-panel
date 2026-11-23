"""Memorization probe (H2): does the LLM already know the answer?

We ask the model, with no evidence at all, for the first-reported U.S.
unemployment rate of each month and score its answer against the ALFRED
first release. If the model is accurate for months published before its
training cutoff and no better than a naive guess afterwards, it has
memorized history, which is why the main evaluation uses post-cutoff months.

A month counts as "pre-cutoff" when its first release came out on or before
the model's documented cutoff (Llama 3.1: December 2023). So the December
2023 value, published January 5, 2024, is post-cutoff.

Comparator: the naive guess is the last value published before the cutoff
(what a model frozen at its cutoff could "know"). A model that has not seen
post-cutoff data should do about as well as this guess after the cutoff and
much better than it before the cutoff.

Backends:
    --backend openai   any OpenAI-compatible server: vLLM, Ollama, LM Studio
                       (default URL http://localhost:11434/v1, i.e. Ollama)
    --backend hf       local transformers model (needs HF_TOKEN for Llama)

Examples:
    ollama pull llama3.1:8b
    python -m src.evaluation.memorization --backend openai --model llama3.1:8b

    python -m src.evaluation.memorization --backend hf \
        --model meta-llama/Llama-3.1-8B-Instruct --load-4bit

Each run writes results/memorization/<run_id>/ with probe.csv, summary.csv
and run.json (model, backend, prompt version, settings).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd

from src.gatekeeper import REPO_ROOT, clean_vintages, from_fred_all_releases

PROMPT_VERSION = "recall-v2"
SYSTEM_PROMPT = "You are a careful assistant. Answer with a single number only."
USER_TEMPLATE = (
    "What was the U.S. unemployment rate (seasonally adjusted, as first reported "
    "by the Bureau of Labor Statistics) for {month}? "
    "Reply with only the percentage as a number with one decimal, for example 4.2. "
    "If you do not know, give your best guess."
)
# Sent once if the first reply contains no number (e.g. "I don't have that information").
FORCE_TEMPLATE = (
    "I understand you may not know for sure. Please give your single best numeric "
    "guess for the U.S. unemployment rate in {month}, as a number with one decimal only."
)

MODEL_CUTOFF = "2023-12-31"  # Llama 3.1 model card: pretraining data cutoff December 2023
PROBE_START = "2022-01-01"
PROBE_END = "2026-09-01"
OUT_ROOT = REPO_ROOT / "results" / "memorization"

_NUMBER = re.compile(r"(?<![\d.])(\d{1,2}(?:\.\d+)?)(?![\d.]*\d)")


# ---- truth -----------------------------------------------------------------
def first_release_unrate(vintages: pd.DataFrame | None = None) -> pd.DataFrame:
    """date, value, first_release for UNRATE (downloads from ALFRED if needed)."""
    if vintages is None:
        from dotenv import load_dotenv
        from fredapi import Fred

        load_dotenv(REPO_ROOT / ".env")
        raw = Fred(api_key=os.getenv("FRED_API_KEY")).get_series_all_releases("UNRATE")
        vintages = from_fred_all_releases("UNRATE", raw)
    df = clean_vintages(vintages)
    df = df[df["series_id"] == "UNRATE"].sort_values(["date", "realtime_start"])
    first = df.groupby("date", as_index=False).first()
    return first.rename(columns={"realtime_start": "first_release"})[["date", "value", "first_release"]]


def probe_months(truth: pd.DataFrame, start=PROBE_START, end=PROBE_END, cutoff=MODEL_CUTOFF) -> pd.DataFrame:
    cut = pd.Timestamp(cutoff)
    months = truth[(truth["date"] >= pd.Timestamp(start)) & (truth["date"] <= pd.Timestamp(end))].copy()
    months["period"] = months["first_release"].le(cut).map({True: "pre_cutoff", False: "post_cutoff"})
    known = truth[truth["first_release"] <= cut]
    months["naive_guess"] = float(known.sort_values("date")["value"].iloc[-1])
    return months.reset_index(drop=True)


# ---- model backends ---------------------------------------------------------
class OpenAICompatible:
    """Minimal chat client for vLLM / Ollama / any /v1/chat/completions server."""

    def __init__(self, model: str, base_url: str, api_key: str | None = None, timeout: int = 600):
        self.model, self.base_url, self.api_key = model, base_url.rstrip("/"), api_key or "none"
        self.timeout = timeout

    def __call__(self, system: str, user: str, temperature: float, max_tokens: int) -> tuple[str, int]:
        import urllib.request

        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        # The first call can be slow while the model loads into memory; retry once on timeout.
        for attempt in range(2):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    out = json.loads(resp.read())
                break
            except TimeoutError:
                if attempt == 1:
                    raise
        text = out["choices"][0]["message"]["content"]
        tokens = int(out.get("usage", {}).get("total_tokens", 0))
        return text, tokens


class HFLocal:
    """Local transformers model. Greedy decoding when temperature is 0."""

    def __init__(self, model: str, load_4bit: bool = False):
        import torch
        from dotenv import load_dotenv
        from transformers import AutoModelForCausalLM, AutoTokenizer

        load_dotenv(REPO_ROOT / ".env")
        token = os.getenv("HF_TOKEN")
        kwargs = {"device_map": "auto", "token": token}
        if load_4bit:
            from transformers import BitsAndBytesConfig  # needs: pip install bitsandbytes

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        else:
            kwargs["torch_dtype"] = torch.bfloat16
        self.tok = AutoTokenizer.from_pretrained(model, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)

    def __call__(self, system: str, user: str, temperature: float, max_tokens: int) -> tuple[str, int]:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        gen = {"max_new_tokens": max_tokens, "pad_token_id": self.tok.eos_token_id}
        gen.update({"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False})
        out = self.model.generate(ids, **gen)
        new = out[0, ids.shape[1]:]
        return self.tok.decode(new, skip_special_tokens=True), int(out.shape[1])


# ---- scoring ---------------------------------------------------------------
def parse_rate(text: str) -> float | None:
    """First plausible unemployment rate (0 to 30) in the reply, else None."""
    for m in _NUMBER.finditer(text or ""):
        val = float(m.group(1))
        if 0 < val < 30:
            return val
    return None


def summarize(probe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, grp in probe.groupby("period"):
        ans = grp.dropna(subset=["answer"])
        err = (ans["answer"] - ans["value"]).abs()
        naive_err = (grp["naive_guess"] - grp["value"]).abs()
        rows.append(
            {
                "period": period,
                "n_queries": len(grp),
                "n_answered": len(ans),
                "decline_rate": round(grp["declined"].mean(), 3) if "declined" in grp else None,
                "model_mae": round(err.mean(), 3) if len(ans) else None,
                "exact_match_rate": round((err < 0.05).mean(), 3) if len(ans) else None,
                "within_0.2_rate": round((err <= 0.2 + 1e-9).mean(), 3) if len(ans) else None,
                "naive_mae": round(naive_err.mean(), 3),
            }
        )
    out = pd.DataFrame(rows)
    order = {"pre_cutoff": 0, "post_cutoff": 1}
    return out.sort_values("period", key=lambda s: s.map(order)).reset_index(drop=True)


# ---- run -------------------------------------------------------------------
def run_probe(client, months: pd.DataFrame, samples: int, temperature: float, max_tokens: int) -> pd.DataFrame:
    """Ask each month; if the model declines, ask once more for a best guess.

    ``declined`` records whether the first reply had no number. ``answer`` is the
    number from the first reply, or from the follow-up if the first declined.
    """
    rows = []
    for rec in months.itertuples(index=False):
        label = rec.date.strftime("%B %Y")
        for s in range(samples):
            t0 = time.perf_counter()
            text, tokens = client(SYSTEM_PROMPT, USER_TEMPLATE.format(month=label), temperature, max_tokens)
            first = parse_rate(text)
            forced_text, answer = "", first
            if first is None:
                forced_text, more = client(SYSTEM_PROMPT, FORCE_TEMPLATE.format(month=label), temperature, max_tokens)
                tokens += more
                answer = parse_rate(forced_text)
            print(f"  {rec.date:%Y-%m} ({rec.period}): truth {rec.value}, model {answer}"
                  f"{' after decline' if first is None else ''}", flush=True)
            rows.append(
                {
                    "month": rec.date.strftime("%Y-%m"),
                    "period": rec.period,
                    "sample": s,
                    "value": rec.value,
                    "first_release": rec.first_release.date(),
                    "naive_guess": rec.naive_guess,
                    "raw_reply": text.strip(),
                    "declined": first is None,
                    "forced_reply": forced_text.strip(),
                    "answer": answer,
                    "tokens": tokens,
                    "seconds": round(time.perf_counter() - t0, 2),
                }
            )
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="LLM memorization probe for UNRATE (H2)")
    p.add_argument("--backend", choices=["openai", "hf"], default="openai")
    p.add_argument("--model", default="llama3.1:8b")
    p.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"))
    p.add_argument("--load-4bit", action="store_true", help="hf backend only: 4-bit quantization (bitsandbytes)")
    p.add_argument("--samples", type=int, default=1, help="answers per month (use >1 with temperature > 0)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--evidence", help="saved evidence table with UNRATE vintages; default downloads from ALFRED")
    p.add_argument("--timeout", type=int, default=600, help="seconds to wait per model reply (openai backend)")
    p.add_argument("--limit", type=int, help="only probe this many months per period (quick smoke test)")
    args = p.parse_args(argv)

    vint = None
    if args.evidence:
        path = Path(args.evidence)
        vint = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    months = probe_months(first_release_unrate(vint))
    if args.limit:
        months = months.groupby("period", group_keys=False).head(args.limit)

    client = (
        OpenAICompatible(args.model, args.base_url, os.getenv("LLM_API_KEY"), args.timeout)
        if args.backend == "openai"
        else HFLocal(args.model, args.load_4bit)
    )
    probe = run_probe(client, months, args.samples, args.temperature, args.max_tokens)
    summary = summarize(probe)

    run_id = f"{pd.Timestamp.now():%Y%m%d-%H%M%S}-{re.sub(r'[^A-Za-z0-9.]+', '-', args.model)}"
    out = OUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=False)
    probe.to_csv(out / "probe.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "model": args.model,
                "backend": args.backend,
                "load_4bit": args.load_4bit,
                "model_cutoff": MODEL_CUTOFF,
                "prompt_version": PROMPT_VERSION,
                "system_prompt": SYSTEM_PROMPT,
                "user_template": USER_TEMPLATE,
                "force_template": FORCE_TEMPLATE,
                "temperature": args.temperature,
                "samples": args.samples,
                "max_tokens": args.max_tokens,
                "probe_window": [PROBE_START, PROBE_END],
                "truth": "ALFRED UNRATE first release",
                "total_tokens": int(probe["tokens"].sum()),
            },
            indent=2,
        )
    )
    print(summary.to_string(index=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
