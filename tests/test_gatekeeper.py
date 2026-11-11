"""Tests for src.gatekeeper.

Offline tests use a small synthetic vintage table. One live test checks real
ALFRED UNRATE data and is skipped without a FRED key or network.
"""

import os

import pandas as pd
import pytest

from src.gatekeeper import (
    Gatekeeper,
    clean_vintages,
    from_fred_all_releases,
    load_gatekeeper,
)


@pytest.fixture
def vintages():
    # UNRATE-like: Aug value first published 4.1, revised to 4.0 a month later.
    rows = [
        ("UNRATE", "2026-07-01", "2026-08-07", 4.1),
        ("UNRATE", "2026-08-01", "2026-09-04", 4.1),
        ("UNRATE", "2026-08-01", "2026-10-02", 4.0),  # revision
        ("UNRATE", "2026-09-01", "2026-10-02", 4.2),
        ("PAYEMS", "2026-08-01", "2026-09-04", 159000),
        ("PAYEMS", "2026-09-01", "2026-10-02", 159150),
    ]
    return pd.DataFrame(rows, columns=["series_id", "date", "realtime_start", "value"])


@pytest.fixture
def gk(vintages):
    return Gatekeeper(vintages)


def test_future_release_is_hidden(gk):
    # Sep 2026 value was released Oct 2: invisible on Sep 20, visible Oct 20.
    assert pd.Timestamp("2026-09-01") not in set(gk.as_of("UNRATE", "2026-09-20")["date"])
    assert gk.latest("UNRATE", "2026-10-20")["date"] == pd.Timestamp("2026-09-01")


def test_release_on_cutoff_day_is_visible(gk):
    # Released 8:30 a.m. on Oct 2; forecast at 5 p.m. Oct 2 may use it.
    assert gk.latest("UNRATE", "2026-10-02")["date"] == pd.Timestamp("2026-09-01")
    assert gk.latest("UNRATE", "2026-10-01")["date"] == pd.Timestamp("2026-08-01")


def test_revisions_respect_cutoff(gk):
    aug = pd.Timestamp("2026-08-01")
    before = gk.as_of("UNRATE", "2026-09-20").set_index("date")["value"]
    after = gk.as_of("UNRATE", "2026-10-20").set_index("date")["value"]
    assert before[aug] == 4.1  # first release
    assert after[aug] == 4.0   # revised value


def test_never_returns_data_published_after_cutoff(gk):
    for cutoff in ["2026-08-06", "2026-08-07", "2026-09-20", "2026-10-20"]:
        for s in gk.available_series():
            out = gk.as_of(s, cutoff)
            assert (out["realtime_start"] <= pd.Timestamp(cutoff)).all()


def test_one_row_per_date(gk):
    out = gk.as_of("UNRATE", "2026-10-20")
    assert out["date"].is_unique


def test_before_any_release_is_empty(gk):
    assert gk.as_of("UNRATE", "2026-01-01").empty
    assert gk.latest("UNRATE", "2026-01-01") is None


def test_unknown_series_raises(gk):
    with pytest.raises(KeyError):
        gk.as_of("NOTASERIES", "2026-10-20")


def test_full_history_not_exposed(gk):
    assert not hasattr(gk, "vintages")


def test_requests_are_logged(gk):
    gk.as_of("UNRATE", "2026-09-20")
    gk.latest("PAYEMS", "2026-10-20")
    assert [r["series_id"] for r in gk.request_log] == ["UNRATE", "PAYEMS"]


def test_clean_vintages_handles_fred_quirks():
    raw = pd.DataFrame(
        {"date": ["2026-08-01", "2026-09-01"], "realtime_start": ["2026-09-04", "2026-10-02"],
         "value": ["4.1", "."]}
    )
    out = from_fred_all_releases("UNRATE", raw)
    assert len(out) == 1 and out["value"].dtype.kind == "f"


def test_missing_columns_raise():
    with pytest.raises(ValueError):
        clean_vintages(pd.DataFrame({"date": [], "value": []}))


@pytest.mark.skipif(not os.getenv("FRED_API_KEY") and not os.path.exists(".env"),
                    reason="needs FRED API key")
def test_live_unrate_release_timing():
    from dotenv import load_dotenv
    from fredapi import Fred

    load_dotenv()
    raw = Fred(api_key=os.getenv("FRED_API_KEY")).get_series_all_releases("UNRATE")
    gk = Gatekeeper(from_fred_all_releases("UNRATE", raw))
    aug = gk.latest("UNRATE", "2026-09-20")
    assert aug is not None
    assert aug["date"] == pd.Timestamp("2026-08-01")
    sep = gk.latest("UNRATE", "2026-10-20")
    assert sep is not None
    assert sep["date"] == pd.Timestamp("2026-09-01")
    assert sep["realtime_start"] == pd.Timestamp("2026-10-02")


# ---- loader ----------------------------------------------------------------
def test_load_gatekeeper_csv(tmp_path, vintages):
    path = tmp_path / "vintages.csv"
    vintages.to_csv(path, index=False)
    gk = load_gatekeeper(path)
    assert gk.available_series() == ["PAYEMS", "UNRATE"]
    latest = gk.latest("UNRATE", "2026-10-20")
    assert latest is not None
    assert latest["value"] == 4.2


def test_load_gatekeeper_parquet(tmp_path, vintages):
    pytest.importorskip("pyarrow")
    path = tmp_path / "vintages.parquet"
    vintages.to_parquet(path, index=False)
    assert load_gatekeeper(path).available_series() == ["PAYEMS", "UNRATE"]


def test_load_gatekeeper_series_filter(tmp_path, vintages):
    path = tmp_path / "vintages.csv"
    vintages.to_csv(path, index=False)
    gk = load_gatekeeper(path, series=["UNRATE"])
    assert gk.available_series() == ["UNRATE"]
    with pytest.raises(KeyError):
        gk.as_of("PAYEMS", "2026-10-20")


def test_load_gatekeeper_unknown_series_raises(tmp_path, vintages):
    path = tmp_path / "vintages.csv"
    vintages.to_csv(path, index=False)
    with pytest.raises(KeyError):
        load_gatekeeper(path, series=["UNRATE", "NOTASERIES"])


def test_load_gatekeeper_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_gatekeeper(tmp_path / "nope.parquet")
