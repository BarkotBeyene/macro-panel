"""Offline tests for src.agents (fake model, synthetic vintages; no GPU or network)."""

import pandas as pd
import pytest

from src.agents import chair, common, evidence, llm, orchestrator
from src.evaluation import faithfulness, runlog
from src.gatekeeper import Gatekeeper


def _vintages():
    rows = []
    months = pd.date_range("2021-01-01", "2024-06-01", freq="MS")
    for i, d in enumerate(months):
        pub = d + pd.offsets.MonthBegin(1) + pd.Timedelta(days=4)  # published early next month
        rows.append(("UNRATE", d, pub, 3.5 + 0.01 * i))
        rows.append(("PAYEMS", d, pub, 150000 + 200 * i))
        rows.append(("JTSJOL", d, pub + pd.Timedelta(days=30), 8000 - 10 * i))
        rows.append(("INDPRO", d, pub + pd.Timedelta(days=10), 100 + 0.1 * i))
        rows.append(("CPIAUCSL", d, pub + pd.Timedelta(days=8), 280 + 0.5 * i))
        rows.append(("FEDFUNDS", d, d + pd.offsets.MonthBegin(1), 5.33))
    for i, d in enumerate(pd.date_range("2021-01-02", "2024-06-29", freq="W-SAT")):
        rows.append(("ICSA", d, d + pd.Timedelta(days=5), 210000 + 50 * i))
        rows.append(("CCSA", d, d + pd.Timedelta(days=12), 1800000 + 100 * i))
    # a later revision of Jan 2024 UNRATE that must stay invisible on 2024-02-20
    rows.append(("UNRATE", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-15"), 9.9))
    return pd.DataFrame(rows, columns=["series_id", "date", "realtime_start", "value"])


@pytest.fixture
def gk():
    return Gatekeeper(_vintages())


@pytest.fixture
def nowcasts():
    return pd.DataFrame({"episode": pd.to_datetime(["2024-02-20", "2024-02-20"]), "system": ["chronos2", "timesfm3"],
                         "forecast": [3.86, 3.84], "lo80": [3.7, 3.7], "hi80": [4.0, 4.0]})


def test_evidence_respects_cutoff(gk, nowcasts):
    av = evidence.episode_evidence(gk, "2024-02-20", "2024-02", nowcasts)
    assert "9.9" not in av["UNRATE"]  # later revision invisible
    assert "latest published value: 3.9 (for 2024-01)" in av["UNRATE"]
    assert set(evidence.COSTS) <= set(av)
    assert "Nowcast for 2024-02: 3.86" in av["CHRONOS2"]


def test_buy_enforces_budget_and_menu(gk):
    av = evidence.episode_evidence(gk, "2024-02-20", "2024-02")
    bought, spent = evidence.buy(["payems", "ICSA", "BOGUS", "ICSA", "CCSA"], av, [], 2)
    assert bought == ["PAYEMS", "ICSA"] and spent == 2
    assert evidence.buy(["CHRONOS2"], av, [], 4) == ([], 0)  # no saved nowcast -> not on the menu


def test_parse_json_tolerates_chatter():
    assert llm.parse_json('Sure! ```json\n{"forecast": 4.1}\n```') == {"forecast": 4.1}
    assert llm.parse_json("no json here") is None


def test_clean_cited_drops_unseen_and_reranks():
    cited = [{"series_id": "icsa", "direction": "UP", "rank": 2}, {"series_id": "PAYEMS", "direction": "down", "rank": 1},
             {"series_id": "GDP", "direction": "up", "rank": 3}]
    out = common.clean_cited(cited, {"PAYEMS", "ICSA", "UNRATE"})
    assert out == [{"series_id": "PAYEMS", "direction": "down", "rank": 1},
                   {"series_id": "ICSA", "direction": "up", "rank": 2}]


def test_panel_record_has_what_faithfulness_needs(gk, nowcasts):
    av = evidence.episode_evidence(gk, "2024-02-20", "2024-02", nowcasts)
    rec = orchestrator.run_panel(llm.FakeChat(), {"episode": "2024-02-20", "target_month": "2024-02"}, av, 0, "fake")
    assert rec["acquired"] == ["UNRATE", "PAYEMS", "ICSA", "CHRONOS2"]
    assert rec["credits"] == 3 and rec["credits"] <= evidence.BUDGET
    assert set(rec["evidence"]) == {"UNRATE", "PAYEMS", "ICSA", "CHRONOS2"}
    assert [c["series_id"] for c in rec["cited"]] == ["PAYEMS", "ICSA"]
    assert rec["debate"]["economist"]["forecast"] is not None
    assert rec["n_calls"] >= 5 and rec["tokens"] > 0 and not rec["decision_failed"]


def test_run_writes_valid_runlog_and_majority(gk, nowcasts, tmp_path):
    eps = pd.DataFrame({"episode": ["2024-02-20", "2024-03-20"], "target_month": ["2024-02", "2024-03"]})
    path = orchestrator.run(orchestrator.SYSTEMS, eps, llm.FakeChat(), "fake", out_dir=tmp_path, gk=gk,
                            nowcasts=nowcasts, log=lambda *_: None)
    df = runlog.read_runs(path)
    assert sorted(df["system"].value_counts().items()) == [("majority_vote", 2), ("panel", 2), ("single_agent", 2)]
    mv = df[df.system == "majority_vote"].iloc[0]
    assert len(mv["members"]) == orchestrator.N_VOTERS
    # resume: nothing left to do, nothing duplicated
    orchestrator.run(orchestrator.SYSTEMS, eps, llm.FakeChat(), "fake", out_dir=tmp_path, gk=gk,
                     nowcasts=nowcasts, log=lambda *_: None)
    assert len(runlog.read_runs(path)) == 6


def test_redecide_plugs_into_faithfulness(gk, nowcasts, monkeypatch, tmp_path):
    monkeypatch.setattr(chair, "_client", llm.FakeChat())
    eps = pd.DataFrame({"episode": ["2024-02-20"], "target_month": ["2024-02"]})
    path = orchestrator.run(["panel", "single_agent"], eps, llm.FakeChat(), "fake", out_dir=tmp_path, gk=gk,
                            nowcasts=nowcasts, log=lambda *_: None)
    records = runlog.read_runs(path).to_dict("records")
    panel = next(r for r in records if r["system"] == "panel")
    assert chair.redecide(panel, panel["evidence"]) == pytest.approx(panel["forecast"])
    res = faithfulness.ablate_record(panel, chair.redecide, n_repeats=2, groups=faithfulness.DEFAULT_GROUPS)
    infl = res.set_index(["condition", "item"])["influence"]
    assert infl[("remove", "PAYEMS")] == pytest.approx(-0.05)  # fake chair: PAYEMS pushed down
    assert infl[("remove", "ICSA")] == pytest.approx(0.05)
    assert infl[("remove", "CHRONOS2")] == pytest.approx(0.0)


def test_select_pilot_episodes():
    eps = pd.DataFrame({"episode": [str(d.date()) for d in pd.date_range("2024-01-20", periods=33, freq="MS")]})
    pilot = orchestrator.select_episodes(eps, ["pilot"])
    assert len(pilot) == 12 and pilot.episode.iloc[0] == eps.episode.iloc[0] and pilot.episode.iloc[-1] == eps.episode.iloc[-1]
