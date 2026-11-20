"""Tests for src.build_evidence and src.build_episodes (offline, no FRED key)."""

import csv

import pandas as pd
import pytest

from src.build_episodes import build_episodes, first_releases
from src.build_evidence import build, update_manifest
from src.gatekeeper import Gatekeeper


class FakeFred:
    """Mimics fredapi.Fred.get_series_all_releases: columns date, realtime_start, value."""

    def __init__(self, data):
        self.data = data
        self.calls = []

    def get_series_all_releases(self, series_id):
        self.calls.append(series_id)
        rows = self.data[series_id]
        return pd.DataFrame(rows, columns=["date", "realtime_start", "value"])


UNRATE = [
    # date, published, value
    ("2025-08-01", "2025-09-05", 4.3),
    ("2025-09-01", "2025-11-20", 4.4),   # late release (shutdown)
    ("2025-10-01", "2025-12-16", "."),   # never published: FRED shows "."
    ("2025-11-01", "2025-12-16", 4.6),
    ("2025-11-01", "2026-01-09", 4.5),   # revision: truth must stay 4.6
    ("2025-12-01", "2026-01-09", 4.4),
]
PAYEMS = [("2025-11-01", "2025-12-16", 159000), ("2025-12-01", "2026-01-09", 159100)]


@pytest.fixture
def built(tmp_path):
    fred = FakeFred({"UNRATE": UNRATE, "PAYEMS": PAYEMS})
    paths = dict(raw_dir=tmp_path / "raw", evidence_path=tmp_path / "vintages.csv",
                 manifest_path=tmp_path / "manifest.csv")
    table = build(["UNRATE", "PAYEMS"], fred=fred, **paths)
    return table, fred, paths


def test_build_writes_raw_files_table_and_manifest(built):
    table, fred, p = built
    assert fred.calls == ["UNRATE", "PAYEMS"]
    assert (p["raw_dir"] / "UNRATE.csv").exists() and (p["raw_dir"] / "PAYEMS.csv").exists()
    assert sorted(table["series_id"].unique()) == ["PAYEMS", "UNRATE"]
    assert list(table.columns) == ["series_id", "date", "realtime_start", "value"]
    # "." is dropped, so the unpublished October 2025 has no row.
    assert pd.Timestamp("2025-10-01") not in set(table.loc[table.series_id == "UNRATE", "date"])
    with p["manifest_path"].open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and all(len(r["sha256"]) == 64 for r in rows)
    assert rows[0]["source_url"].startswith("https://alfred.stlouisfed.org/")


def test_manifest_rerun_replaces_rows_instead_of_duplicating(tmp_path):
    path = tmp_path / "manifest.csv"
    update_manifest([{"filename": "a.csv", "sha256": "1"}], path)
    update_manifest([{"filename": "a.csv", "sha256": "2"}, {"filename": "b.csv", "sha256": "3"}], path)
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert [(r["filename"], r["sha256"]) for r in rows] == [("a.csv", "2"), ("b.csv", "3")]


def test_build_without_key_explains_what_to_do(tmp_path, monkeypatch):
    pytest.importorskip("fredapi")
    pytest.importorskip("dotenv")
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setattr("src.build_evidence.REPO_ROOT", tmp_path)  # no .env there
    with pytest.raises(SystemExit, match="FRED_API_KEY"):
        build(["UNRATE"], raw_dir=tmp_path / "raw", evidence_path=tmp_path / "v.csv",
              manifest_path=tmp_path / "m.csv")


@pytest.fixture
def episodes(built):
    table, _, _ = built
    return build_episodes(Gatekeeper(table), table, start="2025-08", end="2025-12")


def test_truth_is_first_release_not_revision(built):
    table, _, _ = built
    truth = first_releases(table)
    assert truth.loc[pd.Timestamp("2025-11-01"), "truth_first_release"] == 4.6


def test_unpublished_month_is_excluded_with_reason(episodes):
    eps, excluded = episodes
    assert "UNRATE-2025-10" not in set(eps["episode_id"])
    assert list(excluded["reason"]) == ["no first release published"]
    assert len(eps) == 4


def test_answer_is_never_visible_at_cutoff(episodes):
    eps, _ = episodes
    assert (pd.to_datetime(eps["truth_release_date"]) > pd.to_datetime(eps["cutoff"])).all()
    assert (pd.to_datetime(eps["cutoff"]).dt.day == 20).all()


def test_gatekeeper_view_at_cutoff(episodes):
    eps, _ = episodes
    e = eps.set_index("episode_id")
    # On Nov 20, 2025 the late September figure came out that morning: visible.
    assert str(e.loc["UNRATE-2025-11", "last_visible_date"]) == "2025-09-01"
    assert e.loc["UNRATE-2025-11", "gap_months"] == 2
    # On Sep 20, 2025 the latest visible month is August.
    assert str(e.loc["UNRATE-2025-09", "last_visible_date"]) == "2025-08-01"
    assert e.loc["UNRATE-2025-09", "gap_months"] == 1
