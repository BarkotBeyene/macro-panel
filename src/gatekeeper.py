"""Gatekeeper: deterministic, vintage-aware access to economic evidence.

The panel never reads data directly. It asks the Gatekeeper, which answers
"what could a forecaster have seen at this cutoff?" using ALFRED vintages.

Rule: for each observation month, return the value from the latest vintage
whose ``realtime_start`` is on or before the cutoff date. Anything published
after the cutoff is invisible, including later revisions.

Cutoff convention: forecasts are made at 5 p.m. ET on the cutoff date. BLS,
DOL and Fed releases come out in the morning (8:30 a.m. or 9:15 a.m. ET), so a
release dated on the cutoff day counts as available (``<=``, not ``<``).

Input table (the evidence-table contract, one row per vintage of a value):

    series_id        str        e.g. "UNRATE"
    date             datetime   the month/week the value describes
    realtime_start   datetime   the day that value became public
    value            float
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("series_id", "date", "realtime_start", "value")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_PATH = REPO_ROOT / "data" / "processed" / "evidence_tables" / "vintages.parquet"


def _to_timestamp(cutoff) -> pd.Timestamp:
    ts = pd.Timestamp(cutoff)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def clean_vintages(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a vintage table to the evidence-table contract."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"vintage table is missing columns: {missing}")
    out = df.loc[:, list(REQUIRED_COLUMNS)].copy()
    out["series_id"] = out["series_id"].astype(str)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["realtime_start"] = pd.to_datetime(out["realtime_start"]).dt.normalize()
    # ALFRED uses "." for missing values; coerce and drop them.
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"])
    if out.duplicated(["series_id", "date", "realtime_start"]).any():
        raise ValueError("duplicate (series_id, date, realtime_start) rows")
    return out.sort_values(["series_id", "date", "realtime_start"]).reset_index(drop=True)


def from_fred_all_releases(series_id: str, raw: pd.DataFrame) -> pd.DataFrame:
    """Convert fredapi ``get_series_all_releases`` output to the contract."""
    df = raw.copy()
    df["series_id"] = series_id
    return clean_vintages(df)


@dataclass
class Gatekeeper:
    """Answers evidence requests as of a cutoff date and logs every request."""

    vintages: pd.DataFrame
    request_log: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        # Private, validated copy: callers cannot reach the full history.
        self._data = clean_vintages(self.vintages)
        del self.vintages

    # ---- discovery -------------------------------------------------------
    def available_series(self) -> list[str]:
        return sorted(self._data["series_id"].unique())

    # ---- core query ------------------------------------------------------
    def as_of(self, series_id: str, cutoff, start=None) -> pd.DataFrame:
        """History of ``series_id`` as a forecaster saw it on ``cutoff``.

        Returns one row per observation date: date, value, realtime_start
        (the vintage used). Optionally trims to dates >= ``start``.
        """
        if series_id not in self.available_series():
            raise KeyError(f"unknown series: {series_id}")
        cut = _to_timestamp(cutoff)
        rows = self._data[
            (self._data["series_id"] == series_id) & (self._data["realtime_start"] <= cut)
        ]
        # Latest vintage on or before the cutoff, per observation date.
        latest = rows.groupby("date", as_index=False).tail(1)
        if start is not None:
            latest = latest[latest["date"] >= _to_timestamp(start)]
        result = latest.loc[:, ["date", "value", "realtime_start"]].reset_index(drop=True)
        self._log(series_id, cut, len(result), result)
        return result

    def latest(self, series_id: str, cutoff) -> dict | None:
        """Most recent observation visible on ``cutoff`` (or None)."""
        hist = self.as_of(series_id, cutoff)
        if hist.empty:
            return None
        row = hist.iloc[-1]
        return {
            "series_id": series_id,
            "date": row["date"],
            "value": float(row["value"]),
            "realtime_start": row["realtime_start"],
        }

    # ---- audit -----------------------------------------------------------
    def _log(self, series_id, cutoff, n_rows, result) -> None:
        self.request_log.append(
            {
                "series_id": series_id,
                "cutoff": cutoff,
                "n_rows": n_rows,
                "last_date": result["date"].max() if n_rows else None,
                "max_realtime_start": result["realtime_start"].max() if n_rows else None,
            }
        )


# ---- loaders ---------------------------------------------------------------
def load_gatekeeper(path=None, series=None) -> Gatekeeper:
    """Build a Gatekeeper from the saved evidence table.

    ``path`` defaults to data/processed/evidence_tables/vintages.parquet
    (``.csv`` also works). ``series`` optionally restricts which series the
    panel may request; every listed series must be in the table.
    """
    path = Path(path) if path is not None else DEFAULT_EVIDENCE_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"evidence table not found: {path}. Run src/build_evidence.py first, "
            "or use gatekeeper_from_fred() to fetch directly."
        )
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"unsupported file type: {path.suffix} (use .parquet or .csv)")
    if series is not None:
        series = list(series)
        missing = sorted(set(series) - set(df["series_id"].astype(str)))
        if missing:
            raise KeyError(f"series not in evidence table: {missing}")
        df = df[df["series_id"].astype(str).isin(series)]
    return Gatekeeper(df)


def gatekeeper_from_fred(series, api_key=None) -> Gatekeeper:
    """Build a Gatekeeper by downloading ALFRED vintages directly.

    For quick work before the evidence table exists. Experiments should use
    load_gatekeeper() so every run reads the same saved snapshot.
    """
    from dotenv import load_dotenv
    from fredapi import Fred

    load_dotenv(REPO_ROOT / ".env")
    fred = Fred(api_key=api_key or os.getenv("FRED_API_KEY"))
    frames = [from_fred_all_releases(s, fred.get_series_all_releases(s)) for s in series]
    return Gatekeeper(pd.concat(frames, ignore_index=True))
