"""Offline tests for src.evaluation.{runlog, forecasting, costs, faithfulness}."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import costs, faithfulness, forecasting, runlog


@pytest.fixture
def runs(tmp_path):
    recs = []
    actual = {"2024-01": 3.7, "2024-02": 3.9, "2024-03": 3.8, "2024-04": 3.9}
    for m, y in actual.items():
        ep = f"{m}-20"
        recs.append({"episode": ep, "target_month": m, "system": "no_change", "forecast": 3.7, "lo80": 3.5, "hi80": 3.9})
        recs.append({"episode": ep, "target_month": m, "system": "panel", "forecast": y + 0.05, "lo80": y - 0.2,
                     "hi80": y + 0.2, "cited": [{"series_id": "PAYEMS", "direction": "down", "rank": 1}],
                     "acquired": ["PAYEMS", "ICSA"], "credits": 2, "tokens": 1500, "n_calls": 5, "runtime_s": 12.0})
    path = tmp_path / "run" / "runs.jsonl"
    runlog.write_runs(recs, path)
    truth = pd.DataFrame({"target_month": list(actual), "target_first_release": list(actual.values()),
                          "last_known": [3.7, 3.7, 3.9, 3.8]})
    tpath = tmp_path / "episodes.csv"
    truth.to_csv(tpath, index=False)
    return path.parent, tpath


def test_runlog_fills_optional_fields(runs):
    df = runlog.read_runs(runs[0])
    assert len(df) == 8
    nc = df[df.system == "no_change"].iloc[0]
    assert nc.tokens == 0 and nc.cited == [] and nc.acquired == []


def test_runlog_rejects_missing_required():
    with pytest.raises(ValueError):
        runlog.validate_runs(pd.DataFrame({"episode": ["2024-01-20"], "system": ["x"]}))


def test_scores_and_coverage(runs):
    df = forecasting.attach_truth(runlog.read_runs(runs[0]), forecasting.load_truth(runs[1]))
    s = forecasting.score(df).set_index("system")
    assert s.loc["panel", "mae"] == pytest.approx(0.05)
    assert s.loc["no_change", "mae"] == pytest.approx(0.125)
    assert s.loc["panel", "coverage80"] == 1.0
    assert s.loc["no_change", "coverage80"] == 1.0
    assert s.index[0] == "panel"  # sorted by MAE


def test_interval_score_penalizes_misses():
    inside = forecasting.interval_score([4.0], [3.8], [4.2])[0]
    outside = forecasting.interval_score([4.5], [3.8], [4.2])[0]
    assert inside == pytest.approx(0.4) and outside == pytest.approx(0.4 + 10 * 0.3)


def test_compare_to_reference(runs):
    df = forecasting.attach_truth(runlog.read_runs(runs[0]), forecasting.load_truth(runs[1]))
    comp = forecasting.compare_to(df, "no_change", n_boot=200)
    row = comp.iloc[0]
    assert row.system == "panel" and row.mae_diff == pytest.approx(-0.075) and row.ci90_high <= 0


def test_unpublished_month_is_dropped(runs, tmp_path):
    df = runlog.read_runs(runs[0])
    extra = df.iloc[[0]].assign(target_month=pd.Timestamp("2025-10-01"), episode=pd.Timestamp("2025-10-20"))
    scored = forecasting.attach_truth(pd.concat([df, extra]), forecasting.load_truth(runs[1]))
    assert pd.Timestamp("2025-10-01") not in set(scored.target_month)


def test_costs(runs):
    c = costs.summarize_costs(runlog.read_runs(runs[0])).set_index("system")
    assert c.loc["panel", "tokens_per_forecast"] == 1500
    assert c.loc["panel", "evidence_items_per_forecast"] == 2
    assert c.loc["no_change", "llm_calls_per_forecast"] == 0


# ---- faithfulness ------------------------------------------------------------
def fake_chair(record, evidence):
    """Known ground truth: forecast = 4.0, PAYEMS lowers it by 0.3, ICSA raises it by 0.1, JTSJOL does nothing."""
    return 4.0 - 0.3 * ("PAYEMS" in evidence) + 0.1 * ("ICSA" in evidence)


def _record(cited):
    return {"episode": "2024-06-20", "system": "panel", "cited": cited,
            "evidence": {"PAYEMS": "...", "ICSA": "...", "JTSJOL": "..."}}


def test_faithful_explanation_scores_high():
    cited = [{"series_id": "PAYEMS", "direction": "down", "rank": 1}, {"series_id": "ICSA", "direction": "up", "rank": 2}]
    res = faithfulness.ablate_record(_record(cited), fake_chair, n_repeats=1)
    s = faithfulness.summarize_faithfulness(res).set_index("condition").loc["remove"]
    assert s.cited_influence > s.uncited_influence
    assert s.cited_moved_share == 1.0 and s.direction_agreement == 1.0
    assert s.rank_agreement == pytest.approx(1.0)


def test_unfaithful_explanation_is_caught():
    # Chair cites JTSJOL as the main driver and gets PAYEMS's direction wrong.
    cited = [{"series_id": "JTSJOL", "direction": "up", "rank": 1}, {"series_id": "PAYEMS", "direction": "up", "rank": 2}]
    res = faithfulness.ablate_record(_record(cited), fake_chair, n_repeats=1)
    s = faithfulness.summarize_faithfulness(res).set_index("condition").loc["remove"]
    assert s.cited_moved_share == 0.5
    assert s.direction_agreement == 0.0
    assert s.rank_agreement < 0


def test_noise_threshold_and_groups():
    rng = np.random.default_rng(0)

    def noisy(record, evidence):
        return fake_chair(record, evidence) + rng.normal(0, 0.01)

    cited = [{"series_id": "PAYEMS", "direction": "down", "rank": 1}]
    res = faithfulness.ablate_record(_record(cited), noisy, n_repeats=3, groups=faithfulness.DEFAULT_GROUPS)
    group = res[res.condition == "remove_group"].iloc[0]
    assert group.members == "PAYEMS,JTSJOL,ICSA" and group.cited
    assert group.influence == pytest.approx(-0.2, abs=0.05)


def test_replacement_condition():
    def swap(series_id, record):
        return "other month"

    def chair(record, evidence):
        return 4.0 - 0.3 * (evidence.get("PAYEMS") == "...")

    res = faithfulness.ablate_record(_record([]), chair, n_repeats=1, replacement=swap)
    rep = res[(res.condition == "replace") & (res["item"] == "PAYEMS")].iloc[0]
    assert rep.influence == pytest.approx(-0.3)


def test_record_without_evidence_raises():
    with pytest.raises(ValueError):
        faithfulness.ablate_record({"episode": "x", "system": "panel"}, fake_chair)
