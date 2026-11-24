"""Build the evidence table the Gatekeeper reads.

Downloads every release and revision (ALFRED vintages) of each series in
``SERIES`` from the FRED API and writes:

- ``data/raw/alfred/<SERIES_ID>.csv``: the raw download, unchanged
- ``data/processed/evidence_tables/vintages.parquet``: all series in the
  evidence-table contract (series_id, date, realtime_start, value)
- one row per raw file in ``data/raw/manifest.csv`` (source, time, SHA-256)

Run once, commit the outputs, and every experiment then reads the same saved
snapshot through ``load_gatekeeper()``:

    python -m src.build_evidence            # all series in SERIES
    python -m src.build_evidence UNRATE ICSA

Needs FRED_API_KEY in ``.env`` (free key: fred.stlouisfed.org > My Account >
API Keys). Never commit ``.env``.
"""

from __future__ import annotations

import csv
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.gatekeeper import REPO_ROOT, clean_vintages, from_fred_all_releases

RAW_DIR = REPO_ROOT / "data" / "raw" / "alfred"
MANIFEST_PATH = REPO_ROOT / "data" / "raw" / "manifest.csv"
EVIDENCE_PATH = REPO_ROOT / "data" / "processed" / "evidence_tables" / "vintages.parquet"
MANIFEST_COLUMNS = ["filename", "source_url", "downloaded_at", "sha256", "notes"]

# The forecast target plus the evidence the panel may request. Each entry is
# (description, release frequency, publisher). Agents can show these
# descriptions to the LLM; the Gatekeeper serves the values.
TARGET = "UNRATE"
SERIES: dict[str, tuple[str, str, str]] = {
    "UNRATE": ("Unemployment rate, %, seasonally adjusted (forecast target)", "monthly", "BLS"),
    "PAYEMS": ("Total nonfarm payroll employment, thousands", "monthly", "BLS"),
    "ICSA": ("Initial unemployment insurance claims, weekly", "weekly", "DOL"),
    "CCSA": ("Continued unemployment insurance claims, weekly", "weekly", "DOL"),
    "JTSJOL": ("Job openings, total nonfarm (JOLTS), thousands", "monthly", "BLS"),
    "INDPRO": ("Industrial production index", "monthly", "Federal Reserve Board"),
    "CPIAUCSL": ("Consumer price index, all urban consumers", "monthly", "BLS"),
    "FEDFUNDS": ("Effective federal funds rate, %, monthly average", "monthly", "Federal Reserve Board"),
}


def source_url(series_id: str) -> str:
    return f"https://alfred.stlouisfed.org/series?seid={series_id}"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def update_manifest(rows: list[dict], path: Path = MANIFEST_PATH) -> None:
    """Add or replace manifest rows, keyed by filename."""
    existing: dict[str, dict] = {}
    if path.exists() and path.stat().st_size:
        with path.open(newline="") as f:
            existing = {r["filename"]: r for r in csv.DictReader(f)}
    for r in rows:
        existing[r["filename"]] = r
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        for name in sorted(existing):
            w.writerow({k: existing[name].get(k, "") for k in MANIFEST_COLUMNS})


def download_series(fred, series_id: str, raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, dict]:
    """Download all releases of one series; save the raw CSV; return (clean table, manifest row)."""
    raw = fred.get_series_all_releases(series_id)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{series_id}.csv"
    raw.to_csv(path, index=False)
    clean = from_fred_all_releases(series_id, raw)
    row = {
        "filename": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
        "source_url": source_url(series_id),
        "downloaded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256": sha256_of(path),
        "notes": (
            f"{SERIES.get(series_id, ('', '', ''))[0]}; all releases via fredapi "
            f"get_series_all_releases; {len(raw)} raw rows, {len(clean)} usable, "
            f"observations {clean['date'].min():%Y-%m-%d} to {clean['date'].max():%Y-%m-%d}"
        ),
    }
    return clean, row


def build(series_ids=None, fred=None, raw_dir: Path = RAW_DIR, evidence_path: Path = EVIDENCE_PATH,
          manifest_path: Path = MANIFEST_PATH) -> pd.DataFrame:
    """Download every series, write the evidence table and manifest, return the table."""
    series_ids = list(series_ids or SERIES)
    if fred is None:
        from dotenv import load_dotenv
        from fredapi import Fred

        load_dotenv(REPO_ROOT / ".env")
        key = os.getenv("FRED_API_KEY")
        if not key or key == "your_fred_api_key_here":
            raise SystemExit("Set FRED_API_KEY in .env first (copy .env.example to .env).")
        fred = Fred(api_key=key)

    frames, rows = [], []
    for sid in series_ids:
        clean, row = download_series(fred, sid, raw_dir)
        frames.append(clean)
        rows.append(row)
        print(f"{sid:9s} {len(clean):7d} rows  {row['notes'].split('; ')[-1]}")

    table = clean_vintages(pd.concat(frames, ignore_index=True))
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    if evidence_path.suffix == ".parquet":
        table.to_parquet(evidence_path, index=False)
    else:
        table.to_csv(evidence_path, index=False)
    update_manifest(rows, manifest_path)
    print(f"Wrote {len(table)} rows for {len(series_ids)} series to {evidence_path}")
    return table


if __name__ == "__main__":
    build(sys.argv[1:] or None)
