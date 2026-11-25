"""Run the agent systems on forecast episodes and save them in the run-log format.

Systems (all use the same model, evidence menu and budget; see evidence.py):

    panel          Analyst -> Economist -> Skeptic each round; the Chair then buys more
                   evidence or commits. Up to MAX_ROUNDS rounds, BUDGET credits.
    single_agent   one forecaster: buys evidence or commits, same rounds and budget,
                   then the same final-decision prompt without a discussion.
    majority_vote  no debate: three independent single agents (seeds s, s+1, s+2);
                   median forecast and interval, items cited by at least two of them.
                   Its first member is the single_agent record.

Episodes: ``pilot`` (default) is 12 forecast dates spread evenly over the
33-episode window, as planned in the proposal for Milestone II; ``all`` runs
every episode; or pass dates like 2024-03-20.

Usage (Ollama on the same machine):

    python -m src.agents.orchestrator --systems panel single_agent majority_vote --workers 4

    # resume after a dropped connection (skips forecasts already saved)
    python -m src.agents.orchestrator --resume results/forecasts/<run_id>

    # dry run without a model
    python -m src.agents.orchestrator --backend fake --episodes 2024-01-20 2024-06-20

Output: results/forecasts/<run_id>/runs.jsonl (one line per forecast, written as
each finishes) and run.json (settings, prompts version, data hashes).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents import analyst, chair, economist, skeptic
from src.agents.common import PROMPT_VERSION
from src.agents.evidence import BUDGET, COSTS, TARGET, buy, episode_evidence, load_model_nowcasts, menu_text
from src.agents.llm import DEFAULT_MODEL, DEFAULT_URL, Usage, make_client
from src.build_episodes import EPISODES_PATH
from src.build_evidence import EVIDENCE_PATH
from src.gatekeeper import REPO_ROOT, load_gatekeeper

RESULTS_DIR = REPO_ROOT / "results" / "forecasts"
MAX_ROUNDS = 3
N_VOTERS = 3
PILOT_SIZE = 12
SYSTEMS = ("panel", "single_agent", "majority_vote")


# ---- one forecast ---------------------------------------------------------------
def _base_record(ep, system, seed, model):
    return {"episode": ep["episode"], "target_month": ep["target_month"], "system": system,
            "model": model, "prompt_version": PROMPT_VERSION, "seed": seed, "vintage": ep["episode"]}


def _finish(rec, decision, evidence, acquired, credits, usage, t0, fallback):
    point = decision["forecast"]
    failed = point is None
    if failed:
        point = fallback
    rec.update({
        "forecast": round(float(point), 4),
        "lo80": None if failed else decision["lo80"],
        "hi80": None if failed else decision["hi80"],
        "cited": [] if failed else decision["cited"],
        "explanation": decision.get("explanation", ""),
        "acquired": [TARGET] + acquired,
        "evidence": evidence,
        "credits": credits,
        "tokens": usage.tokens,
        "n_calls": usage.calls,
        "runtime_s": round(time.perf_counter() - t0, 2),
        "parse_failures": usage.parse_failures,
        "decision_failed": failed,
        "transcript": usage.transcript,
    })
    return rec


def run_panel(client, ep, available, seed, model):
    t0, usage = time.perf_counter(), Usage()
    args = (client, ep["episode"], ep["target_month"])
    evidence, acquired, credits, rounds, steps, debate = {TARGET: available[TARGET]}, [], 0, 0, [], None
    for rnd in range(1, MAX_ROUNDS + 1):
        rounds = rnd
        menu = menu_text(available, acquired)
        a = analyst.summarize(*args, evidence, usage, seed=seed)
        e = economist.propose(*args, evidence, a, usage, seed=seed)
        s = skeptic.challenge(*args, evidence, e, menu, usage, seed=seed)
        debate = {"round": rnd, "analyst": a, "economist": e, "skeptic": s}
        left = BUDGET - credits
        if rnd == MAX_ROUNDS or left <= 0 or not any(k not in evidence for k in available if k in COSTS):
            break
        step = chair.next_step(*args, evidence, debate, menu, left, usage, seed=seed)
        steps.append(step)
        if step["action"] != "request":
            break
        bought, spent = buy(step["requests"], available, acquired, left)
        if not bought:
            break
        for sid in bought:
            evidence[sid] = available[sid]
        acquired += bought
        credits += spent
    decision = chair.commit(*args, evidence, debate, usage, seed=seed)
    if debate is None:
        raise RuntimeError("panel completed without a debate result")
    fallback = debate["economist"]["forecast"] or _last_rate(available)
    rec = _base_record(ep, "panel", seed, model)
    rec.update({"rounds": rounds, "chair_steps": steps, "debate": debate})
    return _finish(rec, decision, evidence, acquired, credits, usage, t0, fallback)


def run_single(client, ep, available, seed, model):
    t0, usage = time.perf_counter(), Usage()
    args = (client, ep["episode"], ep["target_month"])
    evidence, acquired, credits, steps = {TARGET: available[TARGET]}, [], 0, []
    for _ in range(MAX_ROUNDS):
        left = BUDGET - credits
        if left <= 0:
            break
        step = chair.next_step(*args, evidence, None, menu_text(available, acquired), left, usage, seed=seed,
                               single=True)
        steps.append(step)
        if step["action"] != "request":
            break
        bought, spent = buy(step["requests"], available, acquired, left)
        if not bought:
            break
        for sid in bought:
            evidence[sid] = available[sid]
        acquired += bought
        credits += spent
    decision = chair.commit(*args, evidence, None, usage, seed=seed)
    rec = _base_record(ep, "single_agent", seed, model)
    rec.update({"rounds": len(steps), "chair_steps": steps})
    return _finish(rec, decision, evidence, acquired, credits, usage, t0, _last_rate(available))


def majority(members: list[dict], seed, model) -> dict:
    """No-debate majority vote over independent single agents."""
    ok = [m for m in members if not m["decision_failed"]] or members
    point = statistics.median(m["forecast"] for m in ok)
    los = [m["lo80"] for m in ok if m["lo80"] is not None]
    his = [m["hi80"] for m in ok if m["hi80"] is not None]
    votes: dict[str, list] = {}
    for m in ok:
        for c in m["cited"]:
            votes.setdefault(c["series_id"], []).append(c)
    cited = []
    for sid, cs in votes.items():
        if len(cs) >= 2:
            dirs = [c["direction"] for c in cs]
            cited.append({"series_id": sid, "direction": max(set(dirs), key=dirs.count),
                          "rank": float(np.mean([c["rank"] for c in cs]))})
    cited.sort(key=lambda c: c["rank"])
    for i, c in enumerate(cited, 1):
        c["rank"] = i
    first = members[0]
    rec = _base_record({"episode": first["episode"], "target_month": first["target_month"]}, "majority_vote", seed, model)
    rec.update({
        "forecast": round(float(point), 4),
        "lo80": float(statistics.median(los)) if los else None,
        "hi80": float(statistics.median(his)) if his else None,
        "cited": cited,
        "acquired": sorted({a for m in members for a in m["acquired"]}),
        "credits": sum(m["credits"] for m in members),
        "tokens": sum(m["tokens"] for m in members),
        "n_calls": sum(m["n_calls"] for m in members),
        "runtime_s": round(sum(m["runtime_s"] for m in members), 2),
        "members": [{"seed": m["seed"], "forecast": m["forecast"], "lo80": m["lo80"], "hi80": m["hi80"],
                     "cited": m["cited"], "acquired": m["acquired"]} for m in members],
        "decision_failed": all(m["decision_failed"] for m in members),
    })
    return rec


def _last_rate(available) -> float:
    import re

    m = re.search(r"latest published value: ([\d.]+)", available[TARGET])
    if m is None:
        raise ValueError(f"{TARGET} evidence is missing a latest published value")
    return float(m.group(1))


# ---- episodes -------------------------------------------------------------------
def select_episodes(episodes: pd.DataFrame, which) -> pd.DataFrame:
    if not which or which == ["pilot"]:
        idx = np.unique(np.round(np.linspace(0, len(episodes) - 1, PILOT_SIZE)).astype(int))
        return episodes.iloc[idx].reset_index(drop=True)
    if which == ["all"]:
        return episodes.reset_index(drop=True)
    wanted = {str(pd.Timestamp(w).date()) for w in which}
    return episodes[episodes["episode"].isin(wanted)].reset_index(drop=True)


# ---- runner ---------------------------------------------------------------------
def run(systems, episodes, client, model, seed=0, workers=1, out_dir=None, gk=None, nowcasts=None, log=print):
    gk = gk or load_gatekeeper(EVIDENCE_PATH)
    nowcasts = load_model_nowcasts() if nowcasts is None else nowcasts
    if out_dir is None:
        raise ValueError("out_dir is required")
    out_path = Path(out_dir) / "runs.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["system"], r["episode"]))
    # Evidence texts are built here, on one thread, before any model call.
    eps = [{"episode": str(e.episode), "target_month": str(e.target_month)} for e in episodes.itertuples()]
    available = {ep["episode"]: episode_evidence(gk, ep["episode"], ep["target_month"], nowcasts) for ep in eps}

    jobs = []
    need_single = "single_agent" in systems or "majority_vote" in systems
    for ep in eps:
        if "panel" in systems and ("panel", ep["episode"]) not in done:
            jobs.append(("panel", ep, seed))
        if need_single:
            want_vote = "majority_vote" in systems and ("majority_vote", ep["episode"]) not in done
            want_single = "single_agent" in systems and ("single_agent", ep["episode"]) not in done
            seeds = range(seed, seed + N_VOTERS) if want_vote else ([seed] if want_single else [])
            jobs += [("single", ep, s) for s in seeds]

    lock = threading.Lock()
    members: dict[tuple[str, int], dict] = {}

    def write(rec):
        with lock, out_path.open("a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    def do(job):
        kind, ep, s = job
        fn = run_panel if kind == "panel" else run_single
        return job, fn(client, ep, available[ep["episode"]], s, model)

    log(f"{len(jobs)} agent runs over {len(eps)} episodes ({', '.join(systems)}), {workers} worker(s)")
    finished = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(do, j) for j in jobs]
        for fut in as_completed(futures):
            finished += 1
            try:
                (kind, ep, s), rec = fut.result()
            except Exception as exc:  # one failed call must not lose the rest of the run
                log(f"  [{finished}/{len(jobs)}] FAILED: {type(exc).__name__}: {exc} (rerun with --resume)")
                continue
            if kind == "panel":
                write(rec)
            else:
                with lock:
                    members[(ep["episode"], s)] = rec
                if s == seed and "single_agent" in systems and ("single_agent", ep["episode"]) not in done:
                    write(rec)
                if "majority_vote" in systems and ("majority_vote", ep["episode"]) not in done:
                    group = []
                    for k in range(seed, seed + N_VOTERS):
                        member = members.get((ep["episode"], k))
                        if member is None:
                            break
                        group.append(member)
                    else:
                        vote = majority(group, seed, model)
                        write(vote)
            log(f"  [{finished}/{len(jobs)}] {ep['episode']} {kind} seed={s}: forecast {rec['forecast']} "
                f"credits {rec['credits']} calls {rec['n_calls']} {rec['runtime_s']}s")
    return out_path


def main(argv=None) -> Path:
    p = argparse.ArgumentParser(description="Run the panel, single agent and majority vote")
    p.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=SYSTEMS)
    p.add_argument("--episodes", nargs="*", default=["pilot"], help="pilot (12), all (33), or forecast dates")
    p.add_argument("--backend", default=None, help="openai (Ollama/vLLM), hf, or fake")
    p.add_argument("--model", default=None)
    p.add_argument("--url", default=None)
    p.add_argument("--load-4bit", action="store_true")
    p.add_argument("--workers", type=int, default=1, help="parallel episodes (set OLLAMA_NUM_PARALLEL to match)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", help="existing run folder to continue")
    args = p.parse_args(argv)

    import os

    for k, v in (("MACRO_LLM_BACKEND", args.backend), ("MACRO_LLM_MODEL", args.model), ("MACRO_LLM_URL", args.url)):
        if v:
            os.environ[k] = v  # so chair.redecide uses the same model later in this process
    client = make_client(args.backend, args.model, args.url, args.load_4bit)
    backend = os.getenv("MACRO_LLM_BACKEND", "openai")
    model = os.getenv("MACRO_LLM_MODEL", DEFAULT_MODEL)

    episodes = select_episodes(pd.read_csv(EPISODES_PATH), args.episodes)
    if args.resume:
        out_dir = Path(args.resume)
        meta = json.loads((out_dir / "run.json").read_text())
        args.systems, args.seed = meta["systems"], meta["seed"]
        episodes = select_episodes(pd.read_csv(EPISODES_PATH), ["all"])
        episodes = episodes[episodes["episode"].isin(meta["episodes"])]
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-agents"
        out_dir = RESULTS_DIR / run_id
        out_dir.mkdir(parents=True, exist_ok=False)
        meta = {
            "run_id": run_id,
            "systems": args.systems,
            "episodes": episodes["episode"].tolist(),
            "backend": backend,
            "model": model,
            "url": os.getenv("MACRO_LLM_URL", DEFAULT_URL) if backend == "openai" else None,
            "prompt_version": PROMPT_VERSION,
            "temperature": chair.CHAIR_TEMPERATURE,
            "seed": args.seed,
            "max_rounds": MAX_ROUNDS,
            "budget_credits": BUDGET,
            "costs": COSTS,
            "n_voters": N_VOTERS,
            "evidence_table_sha256": hashlib.sha256(Path(EVIDENCE_PATH).read_bytes()).hexdigest(),
            "episodes_sha256": hashlib.sha256(Path(EPISODES_PATH).read_bytes()).hexdigest(),
            "python": platform.python_version(),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (out_dir / "run.json").write_text(json.dumps(meta, indent=2))

    path = run(args.systems, episodes, client, model, args.seed, args.workers, out_dir)
    print(f"saved -> {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}")
    return out_dir


if __name__ == "__main__":
    main(sys.argv[1:])
