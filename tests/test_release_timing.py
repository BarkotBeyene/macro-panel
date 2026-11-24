"""Offline tests for src.evaluation.release_timing and src.evaluation.memorization."""

import pandas as pd
import pytest

from src.evaluation import memorization as mem
from src.evaluation import release_timing as rt


@pytest.fixture
def vintages():
    rows = [
        # UNRATE: Jul and Aug on first Fridays; Aug revised later (revision must not move first release).
        ("UNRATE", "2026-07-01", "2026-08-07", 4.1),
        ("UNRATE", "2026-08-01", "2026-09-04", 4.1),
        ("UNRATE", "2026-08-01", "2026-10-02", 4.0),
        ("UNRATE", "2026-09-01", "2026-10-02", 4.2),
        # INDPRO: Aug on time (Sep 16), Sep late (Oct 22, after the 20th).
        ("INDPRO", "2026-08-01", "2026-09-16", 103.1),
        ("INDPRO", "2026-09-01", "2026-10-22", 103.4),
        # JTSJOL: Jul released Tue Sep 1 (two months later, on pattern).
        ("JTSJOL", "2026-07-01", "2026-09-01", 7100),
        # ICSA: week ending Sat Sep 12, released Thu Sep 17.
        ("ICSA", "2026-09-12", "2026-09-17", 230000),
    ]
    return pd.DataFrame(rows, columns=["series_id", "date", "realtime_start", "value"])


@pytest.fixture
def first(vintages):
    return rt.first_releases(vintages, start="2026-01-01")


def test_first_release_ignores_revisions(first):
    aug = first[(first.series_id == "UNRATE") & (first.date == "2026-08-01")].iloc[0]
    assert aug.first_release == pd.Timestamp("2026-09-04")
    assert aug.first_value == 4.1


def test_schedule_flags_late_indpro_only(first):
    checks = rt.check_schedule(first)
    flagged = checks[~checks.matches_schedule]
    assert list(flagged.series_id) == ["INDPRO"]
    assert "day 22" in flagged.issue.iloc[0]


def test_weekly_claims_on_thursday_pass(first):
    checks = rt.check_schedule(first)
    assert checks[checks.series_id == "ICSA"].matches_schedule.all()


def test_late_release_detected_on_forecast_date(first):
    dates = [pd.Timestamp("2026-09-20"), pd.Timestamp("2026-10-20")]
    late = rt.late_releases(first, dates)
    indpro = late[late.series_id == "INDPRO"]
    assert list(indpro.reference_month) == ["2026-09"]
    # UNRATE Sep was out Oct 2, so it is not late on Oct 20.
    assert not ((late.series_id == "UNRATE") & (late.reference_month == "2026-09")).any()


def test_visibility_matches_gatekeeper_rule(first):
    vis = rt.visibility_on_forecast_dates(first, [pd.Timestamp("2026-09-20"), pd.Timestamp("2026-10-20")])
    unrate = vis[vis.series_id == "UNRATE"].set_index("forecast_date")
    assert str(unrate.loc[pd.Timestamp("2026-09-20").date(), "newest_visible"]) == "2026-08-01"
    assert str(unrate.loc[pd.Timestamp("2026-10-20").date(), "newest_visible"]) == "2026-09-01"


def test_release_on_forecast_day_counts_as_visible():
    df = pd.DataFrame([("INDPRO", "2026-08-01", "2026-09-20", 1.0)],
                      columns=["series_id", "date", "realtime_start", "value"])
    first = rt.first_releases(df, start="2026-01-01")
    assert rt.late_releases(first, [pd.Timestamp("2026-09-20")]).empty


def test_write_doc_replaces_only_generated_block(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text(f"intro\n{rt.BEGIN_MARK}\nold\n{rt.END_MARK}\noutro\n")
    rt.write_doc("NEW", doc)
    text = doc.read_text()
    assert "NEW" in text and "old" not in text
    assert text.startswith("intro") and text.rstrip().endswith("outro")


# ---- memorization ------------------------------------------------------------
@pytest.mark.parametrize(
    "reply,expected",
    [("4.2", 4.2), ("The rate was 3.7%.", 3.7), ("I think about 4", 4.0), ("2024: 4.1", 4.1), ("unknown", None)],
)
def test_parse_rate(reply, expected):
    assert mem.parse_rate(reply) == expected


def test_probe_labels_by_release_date_not_reference_month():
    truth = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-11-01", "2023-12-01", "2024-01-01"]),
            "value": [3.7, 3.7, 3.7],
            "first_release": pd.to_datetime(["2023-12-08", "2024-01-05", "2024-02-02"]),
        }
    )
    months = mem.probe_months(truth, start="2023-11-01", end="2024-01-01")
    assert list(months.period) == ["pre_cutoff", "post_cutoff", "post_cutoff"]
    assert (months.naive_guess == 3.7).all()


def test_run_and_summarize_with_fake_model():
    truth = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-10-01", "2024-06-01"]),
            "value": [3.9, 4.1],
            "first_release": pd.to_datetime(["2023-11-03", "2024-07-05"]),
        }
    )
    months = mem.probe_months(truth, start="2023-01-01", end="2024-12-01")
    answers = {"October 2023": "3.9", "June 2024": "3.8"}

    def fake(system, user, temperature, max_tokens):
        month = next(k for k in answers if k in user)
        return answers[month], 10

    probe = mem.run_probe(fake, months, samples=1, temperature=0.0, max_tokens=8)
    summary = mem.summarize(probe).set_index("period")
    assert summary.loc["pre_cutoff", "exact_match_rate"] == 1.0
    assert summary.loc["post_cutoff", "model_mae"] == pytest.approx(0.3)


def test_declined_reply_gets_forced_guess():
    truth = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-10-01", "2024-06-01"]),
            "value": [3.9, 4.1],
            "first_release": pd.to_datetime(["2023-11-03", "2024-07-05"]),
        }
    )
    months = mem.probe_months(truth, start="2024-01-01", end="2024-12-01")
    calls = []

    def fake(system, user, temperature, max_tokens):
        calls.append(user)
        return ("I don't have that information.", 5) if len(calls) == 1 else ("3.9", 5)

    probe = mem.run_probe(fake, months, samples=1, temperature=0.0, max_tokens=8)
    assert bool(probe.declined.iloc[0]) and probe.answer.iloc[0] == 3.9 and probe.tokens.iloc[0] == 10
    assert mem.summarize(probe).decline_rate.iloc[0] == 1.0
