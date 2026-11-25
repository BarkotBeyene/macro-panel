"""Milestone II pilot report: H1 and H3 tables plus a qualitative example, from saved runs.

Reads the agent run, the baseline runs and the faithfulness results, restricts
every system to the same pilot episodes, and writes:

    results/milestone2/<agent_run_id>/
        scores_pilot.csv        accuracy and intervals, all systems, pilot episodes
        comparison_pilot.csv    MAE vs no-change, paired bootstrap 90% CI
        costs_pilot.csv         credits, calls, tokens, runtime
        faithfulness_all.csv    H1 summary, every evidence item
        faithfulness_bought.csv H1 summary, purchased evidence only (UNRATE history left out)
        qualitative.md          one panel record: explanation, citations, removal effects
        summary.md              the tables above, formatted for the write-up
        forecasts_pilot.png     forecasts vs first release (if matplotlib is installed)

Usage:
    python -m src.agents.report results/forecasts/<agent_run> \
        --faithfulness results/faithfulness/<run_id>-panel results/faithfulness/<run_id>-single_agent
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import pandas as pd

from src.build_episodes import EPISODES_PATH
from src.evaluation.costs import summarize_costs
from src.evaluation.faithfulness import summarize_faithfulness
from src.evaluation.forecasting import attach_truth, compare_to, load_truth, score
from src.evaluation.runlog import read_runs
from src.gatekeeper import REPO_ROOT

BASELINES = sorted((REPO_ROOT / "results" / "forecasts").glob("*-baselines"))
ORDER = ["no_change", "ar1", "chronos2", "timesfm3", "single_agent", "majority_vote", "panel"]
NAMES = {"no_change": "No change", "ar1": "AR(1)", "chronos2": "Chronos-2", "timesfm3": "TimesFM-3",
         "single_agent": "Single agent", "majority_vote": "Majority vote (no debate)", "panel": "Panel"}


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else (f"{v:.3f}" if isinstance(v, float) else str(v))
                                       for v in r) + " |")
    return "\n".join(lines)


def _order(df, col="system"):
    df = df.copy()
    df["_o"] = df[col].map({s: i for i, s in enumerate(ORDER)}).fillna(99)
    return df.sort_values("_o").drop(columns="_o")


def h3_table(runs: pd.DataFrame, truth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = attach_truth(runs, truth)
    scores = _order(score(df))
    comp = _order(compare_to(df, "no_change")) if "no_change" in set(df.system) else pd.DataFrame()
    costs = _order(summarize_costs(runs))
    return scores, comp, costs


def doc_h3(scores, costs) -> pd.DataFrame:
    m = scores.merge(costs, on="system", how="left")
    out = pd.DataFrame({
        "System": m["system"].map(NAMES),
        "Episodes": m["n"],
        "MAE": m["mae"], "RMSE": m["rmse"],
        "80% coverage": (m["coverage80"] * 100).round(0).astype("Int64").astype(str) + "%",
        "Interval width": m["width80"],
        "Credits": m["credits_per_forecast"],
        "Tokens per forecast": m["tokens_per_forecast"].round(0).astype("Int64"),
    })
    return out


def doc_h1(summary: pd.DataFrame) -> pd.DataFrame:
    s = summary[summary["condition"] == "remove"]
    s = _order(s)
    pct = lambda x: "" if pd.isna(x) else f"{100 * x:.0f}%"
    return pd.DataFrame({
        "System": s["system"].map(NAMES), "Episodes": s["episodes"], "Cited items": s["cited_items"],
        "Mean influence, cited": s["cited_influence"], "Mean influence, uncited": s["uncited_influence"],
        "Cited items that moved the forecast": s["cited_moved_share"].map(pct),
        "Moved it in the claimed direction": s["direction_agreement"].map(pct),
        "Rank agreement": s["rank_agreement"],
    })


def qualitative(runs: pd.DataFrame, ablations: pd.DataFrame | None, min_effect=0.05) -> str:
    panel = runs[runs["system"] == "panel"]
    if panel.empty:
        return "No panel records."
    pick, why = panel.iloc[0], "first pilot episode"
    if ablations is not None and not ablations.empty:
        a = ablations[(ablations.system == "panel") & (ablations.condition == "remove") & (ablations.item != "UNRATE")]
        a = a.assign(moved=a["influence"].abs() > a["noise_sd"].fillna(0).mul(2).clip(lower=min_effect))
        # Prefer an episode where a cited item did not move the forecast (an unfaithful citation).
        bad = a[a.cited & ~a.moved]
        good = a[a.cited & a.moved]
        target = bad if len(bad) else good
        if len(target):
            ep = target.iloc[0]["episode"]
            pick = panel[panel.episode.dt.strftime("%Y-%m-%d") == ep].iloc[0]
            why = ("a cited item whose removal did not move the forecast" if len(bad)
                   else "cited evidence that did move the forecast")
    lines = [f"### Panel, forecast date {pick.episode:%Y-%m-%d} (target {pick.target_month:%B %Y})",
             f"Chosen because it shows {why}.", "",
             f"- Final forecast: {pick.forecast:.2f} (80% interval {pick.lo80} to {pick.hi80}); "
             f"evidence bought: {', '.join(pick.acquired[1:]) or 'none'}; credits {pick.credits}",
             f"- Chair's explanation: \"{pick.get('explanation', '')}\"",
             "- Cited: " + (", ".join(f"{c['rank']}. {c['series_id']} ({c['direction']})" for c in pick.cited) or "none")]
    debate = pick.get("debate")
    if isinstance(debate, dict):
        lines += [f"- Economist (final round): {debate['economist'].get('forecast')} — {debate['economist'].get('argument', '')}",
                  f"- Skeptic: {debate['skeptic'].get('critique', '')}"]
    if ablations is not None and not ablations.empty:
        a = ablations[(ablations.system == "panel") & (ablations.episode == f"{pick.episode:%Y-%m-%d}")]
        lines += ["", "| Condition | Item | Cited | Claimed direction | Forecast without it | Influence (pp) |",
                  "|---|---|---|---|---|---|"]
        for _, r in a.iterrows():
            lines.append(f"| {r.condition} | {r['item']} | {'yes' if r.cited else 'no'} | {r.claimed_direction or ''} | "
                         f"{r.ablated_forecast:.2f} | {r.influence:+.3f} |")
    return "\n".join(lines)


def plot(runs: pd.DataFrame, truth: pd.DataFrame, path: Path) -> bool:
    try:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError:
        return False
    df = attach_truth(runs, truth)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    act = df.drop_duplicates("target_month").sort_values("target_month")
    ax.plot(act.target_month, act.actual, color="black", lw=2, marker="o", label="First release (truth)")
    styles = {"no_change": ("#8a8a85", "--"), "panel": ("#2a78d6", "-"), "single_agent": ("#eb6834", "-"),
              "majority_vote": ("#1baf7a", ":"), "chronos2": ("#eda100", "-.")}
    for system, (c, ls) in styles.items():
        g = df[df.system == system].sort_values("target_month")
        if len(g):
            ax.plot(g.target_month, g.forecast, color=c, ls=ls, lw=2, marker="o", ms=4, label=NAMES[system])
    ax.set_ylabel("Unemployment rate, %")
    ax.set_title("Pilot nowcasts vs first release (12 episodes)")
    ax.legend(fontsize=7, ncol=3, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(str(path), dpi=200)
    plt.close(fig)
    return True


def main(argv=None) -> Path:
    p = argparse.ArgumentParser(description="Milestone II pilot report")
    p.add_argument("agent_run")
    p.add_argument("--faithfulness", nargs="*", default=[], help="results/faithfulness/<run_id> folder(s)")
    p.add_argument("--baselines", nargs="*", default=[str(b) for b in BASELINES])
    p.add_argument("--out")
    args = p.parse_args(argv)

    agent_dir = Path(args.agent_run)
    agents = read_runs(agent_dir)
    pilot = set(agents["episode"])
    runs = pd.concat([read_runs(*args.baselines), agents], ignore_index=True) if args.baselines else agents
    runs = runs[runs["episode"].isin(pilot)].reset_index(drop=True)
    truth = load_truth(EPISODES_PATH)
    out = Path(args.out) if args.out else REPO_ROOT / "results" / "milestone2" / agent_dir.name
    out.mkdir(parents=True, exist_ok=True)

    scores, comp, costs = h3_table(runs, truth)
    scores.to_csv(out / "scores_pilot.csv", index=False)
    comp.to_csv(out / "comparison_pilot.csv", index=False)
    costs.to_csv(out / "costs_pilot.csv", index=False)

    ablations, h1_all, h1_bought = None, pd.DataFrame(), pd.DataFrame()
    if args.faithfulness:
        ablations = pd.concat([pd.read_csv(Path(f) / "ablations.csv") for f in args.faithfulness], ignore_index=True)
        h1_all = summarize_faithfulness(ablations)
        h1_bought = summarize_faithfulness(ablations[ablations["item"] != "UNRATE"])
        h1_all.to_csv(out / "faithfulness_all.csv", index=False)
        h1_bought.to_csv(out / "faithfulness_bought.csv", index=False)

    fails = agents.groupby("system")["decision_failed"].sum().to_dict() if "decision_failed" in agents else {}
    qual = qualitative(agents, ablations)
    (out / "qualitative.md").write_text(qual + "\n")
    has_plot = plot(runs, truth, out / "forecasts_pilot.png")

    parts = [f"# Milestone II pilot report: {agent_dir.name}", "",
             f"Episodes: {len(pilot)} ({', '.join(sorted(f'{e:%Y-%m}' for e in pilot))}).",
             f"Final decisions with no usable number (fell back to the last published rate): {fails}.", "",
             "## H3: accuracy, intervals and cost (pilot episodes, all systems)", "", _md(doc_h3(scores, costs)), "",
             "MAE difference vs no change (negative = better), paired bootstrap 90% CI:", "", _md(comp), ""]
    if len(h1_all):
        parts += ["## H1: faithfulness, every evidence item", "", _md(doc_h1(h1_all)), "",
                  "## H1: faithfulness, purchased evidence only (unemployment history left out)", "",
                  _md(doc_h1(h1_bought)), "", "Group removal (all labor-market series at once):", "",
                  _md(h1_all[h1_all.condition == "remove_group"]), ""]
    parts += ["## Qualitative example", "", qual, ""]
    if has_plot:
        parts += ["![Pilot nowcasts](forecasts_pilot.png)", ""]
    (out / "summary.md").write_text("\n".join(parts))
    (out / "run.json").write_text(json.dumps(vars(args), indent=2))
    print("\n".join(parts))
    print(f"\nwrote {out}")
    return out


if __name__ == "__main__":
    main()
