"""Release-timing check for the evidence series.

Two questions, both answered from ALFRED's ``realtime_start`` dates:

1. Do ALFRED's first-release dates match the official release schedules?
   (BLS Employment Situation for UNRATE/PAYEMS, BLS JOLTS for JTSJOL,
   Federal Reserve G.17 for INDPRO, DOL weekly claims for ICSA.)
   Every first release is checked against the schedule's usual pattern
   (weekday and day-of-month window). Anything off-pattern is flagged for a
   manual check against the official calendar; it is not silently "fixed".

2. Which releases come out after the 20th? Forecasts are made on the 20th at
   5 p.m. ET. If last month's value for a series is released after the 20th,
   the panel only sees the month before that, so the series is effectively
   one month staler on that forecast date.

Usage (needs FRED_API_KEY in .env, or a saved evidence table):

    python -m src.evaluation.release_timing
    python -m src.evaluation.release_timing --evidence data/processed/evidence_tables/vintages.parquet

Writes CSVs to results/release_timing/ and fills the results block in
docs/release_timing.md.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.gatekeeper import REPO_ROOT, clean_vintages, from_fred_all_releases

SERIES = ("UNRATE", "PAYEMS", "JTSJOL", "INDPRO", "ICSA")

# Window the experiments use (forecast dates Jan 2024 to Sep 2026). We check
# releases from 2023 on so the first forecasts are covered too.
CHECK_START = "2023-01-01"
FORECAST_START = "2024-01-20"
FORECAST_END = "2026-09-20"
FORECAST_DAY = 20

DOC_PATH = REPO_ROOT / "docs" / "release_timing.md"
OUT_DIR = REPO_ROOT / "results" / "release_timing"
BEGIN_MARK = "<!-- BEGIN GENERATED RESULTS -->"
END_MARK = "<!-- END GENERATED RESULTS -->"

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@dataclass(frozen=True)
class Schedule:
    """Usual pattern of an official release, used to flag outliers."""

    source: str
    release: str
    time_et: str
    weekdays: tuple[int, ...]  # allowed weekdays, Mon=0
    day_min: int  # allowed day-of-month window for the release
    day_max: int
    months_after: int | None  # release month minus reference month (None = weekly)
    lag_days: tuple[int, int] | None = None  # if set: days from end of reference month, replaces the month/day checks


SCHEDULES = {
    # First Friday of the following month, occasionally the second Friday.
    "UNRATE": Schedule("BLS", "Employment Situation", "8:30 a.m.", (4,), 1, 12, 1),
    "PAYEMS": Schedule("BLS", "Employment Situation", "8:30 a.m.", (4,), 1, 12, 1),
    # About four to six weeks after the month ends, usually a Tuesday (sometimes
    # Wed to Fri around holidays), which can fall late in the following month.
    "JTSJOL": Schedule("BLS", "JOLTS", "10:00 a.m.", (1, 2, 3, 4), 1, 31, 2, lag_days=(25, 45)),
    # Mid-month for the previous month.
    "INDPRO": Schedule("Federal Reserve Board", "G.17 Industrial Production", "9:15 a.m.", (0, 1, 2, 3, 4), 13, 19, 1),
    # Every Thursday (Wednesday in holiday weeks) for the week ending the prior Saturday.
    "ICSA": Schedule("DOL", "Unemployment Insurance Weekly Claims", "8:30 a.m.", (2, 3), 1, 31, None),
}


# ---- data ------------------------------------------------------------------
def load_vintages(evidence_path=None, series=SERIES) -> pd.DataFrame:
    """Saved evidence table if given, otherwise download from ALFRED."""
    if evidence_path is not None:
        path = Path(evidence_path)
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        df = df[df["series_id"].astype(str).isin(series)]
        return clean_vintages(df)

    import os

    from dotenv import load_dotenv
    from fredapi import Fred

    load_dotenv(REPO_ROOT / ".env")
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    frames = [from_fred_all_releases(s, fred.get_series_all_releases(s)) for s in series]
    return pd.concat(frames, ignore_index=True)


def first_releases(vintages: pd.DataFrame, start=CHECK_START) -> pd.DataFrame:
    """One row per (series, observation date): the day it was first published."""
    df = clean_vintages(vintages)
    first = (
        df.groupby(["series_id", "date"], as_index=False)
        .agg(first_release=("realtime_start", "min"), first_value=("value", "first"))
    )
    first = first[first["date"] >= pd.Timestamp(start)].copy()
    first["release_weekday"] = first["first_release"].dt.day_of_week.map(lambda d: WEEKDAYS[d])
    first["release_day"] = first["first_release"].dt.day
    first["lag_days"] = (first["first_release"] - first["date"]).dt.days
    return first.reset_index(drop=True)


# ---- question 1: schedule check -------------------------------------------
def _months_between(later: pd.Timestamp, earlier: pd.Timestamp) -> int:
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


def check_schedule(first: pd.DataFrame) -> pd.DataFrame:
    """Flag first releases that break their schedule's usual pattern."""
    rows = []
    for rec in first.itertuples(index=False):
        sched = SCHEDULES.get(rec.series_id)
        if sched is None:
            continue
        rel, obs = rec.first_release, rec.date
        issues = []
        if rel.day_of_week not in sched.weekdays:
            issues.append(f"weekday {WEEKDAYS[rel.day_of_week]}")
        if sched.lag_days is not None:
            lag = (rel - (obs + pd.offsets.MonthEnd(0))).days
            lo, hi = sched.lag_days
            if not lo <= lag <= hi:
                issues.append(f"{lag} days after the reference month ended, expected {lo}-{hi}")
        elif sched.months_after is not None:
            gap = _months_between(rel, obs)
            if gap != sched.months_after:
                issues.append(f"released {gap} month(s) after reference month, expected {sched.months_after}")
            elif not (sched.day_min <= rel.day <= sched.day_max):
                issues.append(f"day {rel.day} outside usual {sched.day_min}-{sched.day_max}")
        else:
            # Weekly claims: week ends Saturday, released the following Thu (Wed in holiday weeks).
            if not 3 <= (rel - obs).days <= 6:
                issues.append(f"{(rel - obs).days} days after week end, expected 4-5")
        rows.append(
            {
                "series_id": rec.series_id,
                "date": obs.date(),
                "first_release": rel.date(),
                "weekday": WEEKDAYS[rel.day_of_week],
                "matches_schedule": not issues,
                "issue": "; ".join(issues),
            }
        )
    return pd.DataFrame(rows)


# ---- question 2: what is visible on the 20th -------------------------------
def forecast_dates(start=FORECAST_START, end=FORECAST_END) -> list[pd.Timestamp]:
    months = pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS")
    return [m.replace(day=FORECAST_DAY) for m in months]


def visibility_on_forecast_dates(first: pd.DataFrame, dates=None) -> pd.DataFrame:
    """For each forecast date and series: newest observation already published.

    ``months_stale`` counts whole months between the forecast month and that
    observation (1 = last month is visible). For weekly ICSA it is in days.
    """
    dates = dates or forecast_dates()
    rows = []
    for fdate in dates:
        for sid, grp in first.groupby("series_id"):
            seen = grp[grp["first_release"] <= fdate]
            if seen.empty:
                continue
            newest = seen.loc[seen["date"].idxmax()]
            row = {
                "forecast_date": fdate.date(),
                "series_id": sid,
                "newest_visible": newest["date"].date(),
                "released_on": newest["first_release"].date(),
            }
            if SCHEDULES.get(sid) and SCHEDULES[sid].months_after is None:
                row["staleness"] = f"{(fdate - newest['date']).days} days"
                row["months_stale"] = None
            else:
                stale = _months_between(fdate, newest["date"])
                row["months_stale"] = stale
                row["staleness"] = f"{stale} month(s)"
            rows.append(row)
    return pd.DataFrame(rows)


def late_releases(first: pd.DataFrame, dates=None) -> pd.DataFrame:
    """Monthly releases that a forecaster expected by the 20th but did not get.

    For a forecast on the 20th of month M, the "expected" observation is
    M minus the schedule's usual lag (UNRATE/PAYEMS/INDPRO: M-1, JTSJOL: M-2).
    If that value came out after the 20th, it is listed here.
    """
    dates = dates or forecast_dates()
    present = set(first["series_id"])
    rows = []
    for fdate in dates:
        for sid, sched in SCHEDULES.items():
            if sched.months_after is None or sid not in present:
                continue
            ref = (fdate.replace(day=1) - pd.DateOffset(months=sched.months_after)).normalize()
            hit = first[(first["series_id"] == sid) & (first["date"] == ref)]
            if hit.empty:
                rows.append({"forecast_date": fdate.date(), "series_id": sid,
                             "reference_month": ref.strftime("%Y-%m"), "first_release": None,
                             "note": "never published (no value in ALFRED)"})
                continue
            rel = hit["first_release"].iloc[0]
            if rel > fdate:
                rows.append({"forecast_date": fdate.date(), "series_id": sid,
                             "reference_month": ref.strftime("%Y-%m"), "first_release": rel.date(),
                             "note": f"released {(rel - fdate).days} days after the forecast date"})
    return pd.DataFrame(rows, columns=["forecast_date", "series_id", "reference_month", "first_release", "note"])


# ---- report ----------------------------------------------------------------
def summary_table(first: pd.DataFrame, checks: pd.DataFrame, late: pd.DataFrame) -> pd.DataFrame:
    rows = []
    in_window = first[first["first_release"].between(pd.Timestamp(CHECK_START), pd.Timestamp(FORECAST_END))]
    for sid, sched in SCHEDULES.items():
        grp = in_window[in_window["series_id"] == sid]
        chk = checks[
            (checks["series_id"] == sid)
            & pd.to_datetime(checks["first_release"]).between(pd.Timestamp(CHECK_START), pd.Timestamp(FORECAST_END))
        ]
        if grp.empty:
            continue
        rows.append(
            {
                "Series": sid,
                "Official release": f"{sched.source} {sched.release}, {sched.time_et} ET",
                "Releases checked": len(grp),
                "Match schedule": int(chk["matches_schedule"].sum()),
                "Flagged": int((~chk["matches_schedule"]).sum()),
                "Median release day": int(grp["release_day"].median()) if sched.months_after else "weekly",
                "Releases after the 20th": int((grp["release_day"] > FORECAST_DAY).sum()) if sched.months_after else "n/a",
                "Forecast dates missing expected month": int((late["series_id"] == sid).sum()) if not late.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def _md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_None._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines)


def render_results(summary, checks, late, visibility, run_note) -> str:
    flagged = checks[~checks["matches_schedule"]] if not checks.empty else checks
    stale = (
        visibility.dropna(subset=["months_stale"])
        .groupby("series_id")["months_stale"]
        .agg(["min", "median", "max"])
        .reset_index()
        .rename(columns={"series_id": "Series", "min": "Min months stale", "median": "Median", "max": "Max"})
    )
    parts = [
        f"_{run_note}_",
        "### Summary",
        _md(summary),
        "### What the panel can see on the 20th (monthly series)",
        _md(stale),
        "### Forecast dates where the expected month was not yet published",
        _md(late),
        "### Releases flagged for manual check against the official calendar",
        _md(flagged.drop(columns=["matches_schedule"]) if not flagged.empty else flagged),
    ]
    return "\n\n".join(parts)


def write_doc(results_md: str, doc_path=DOC_PATH) -> None:
    text = Path(doc_path).read_text()
    if BEGIN_MARK not in text or END_MARK not in text:
        raise ValueError(f"{doc_path} is missing the generated-results markers")
    head, rest = text.split(BEGIN_MARK, 1)
    _, tail = rest.split(END_MARK, 1)
    Path(doc_path).write_text(f"{head}{BEGIN_MARK}\n\n{results_md}\n\n{END_MARK}{tail}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--evidence", help="saved evidence table (.parquet or .csv); default downloads from ALFRED")
    parser.add_argument("--no-doc", action="store_true", help="only write CSVs, leave docs/release_timing.md alone")
    args = parser.parse_args(argv)

    vintages = load_vintages(args.evidence)
    first = first_releases(vintages)
    checks = check_schedule(first)
    late = late_releases(first)
    visibility = visibility_on_forecast_dates(first)
    summary = summary_table(first, checks, late)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    first.to_csv(OUT_DIR / "first_releases.csv", index=False)
    checks.to_csv(OUT_DIR / "schedule_check.csv", index=False)
    late.to_csv(OUT_DIR / "late_releases.csv", index=False)
    visibility.to_csv(OUT_DIR / "visibility_on_forecast_dates.csv", index=False)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    source = args.evidence or "ALFRED via fredapi"
    note = f"Generated {pd.Timestamp.now():%Y-%m-%d %H:%M} from {source}; releases from {CHECK_START}, forecast dates {FORECAST_START} to {FORECAST_END}."
    if not args.no_doc:
        write_doc(render_results(summary, checks, late, visibility, note))
    print(summary.to_string(index=False))
    print(f"\nFlagged releases: {int((~checks['matches_schedule']).sum())}  |  Late on forecast dates: {len(late)}")
    print(f"CSVs in {OUT_DIR}")


if __name__ == "__main__":
    main()
