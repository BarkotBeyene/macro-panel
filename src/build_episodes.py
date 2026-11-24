"""Build the forecast episodes (the "exam questions") from the evidence table.

One episode per target month M:

- cutoff: the 20th of month M, 5 p.m. ET (the Gatekeeper treats a release
  dated on the cutoff day as visible)
- target: UNRATE for month M, a nowcast; BLS first publishes it in early M+1
- truth: the first official release of UNRATE for month M, and its date

Columns used by the scoring code (src/evaluation, docs/run_log_format.md):
``episode`` (forecast date), ``target_month`` ("YYYY-MM"),
``target_first_release`` (truth) and ``last_known`` (latest UNRATE visible on
the forecast date). ``truth_first_release`` and ``last_visible_value`` repeat
them under the original names.

Months that were never published (October 2025, government shutdown) have no
truth and are listed as excluded instead of silently dropped.

    python -m src.build_episodes                      # Jan 2024 through the latest scored month
    python -m src.build_episodes 2024-01 2026-09
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.build_evidence import EVIDENCE_PATH, TARGET
from src.gatekeeper import REPO_ROOT, Gatekeeper, load_gatekeeper

EPISODES_PATH = REPO_ROOT / "data" / "processed" / "forecast_episodes" / "episodes.csv"
EXCLUDED_PATH = REPO_ROOT / "data" / "processed" / "forecast_episodes" / "excluded.csv"
CUTOFF_DAY = 20
DEFAULT_START = "2024-01"  # after Llama 3.1's December 2023 training cutoff


def first_releases(vintages: pd.DataFrame, series_id: str = TARGET) -> pd.DataFrame:
    """First published value and its release date for every observation month."""
    s = vintages[vintages["series_id"] == series_id].sort_values(["date", "realtime_start"])
    first = s.groupby("date", as_index=False).head(1)
    return first.rename(columns={"value": "truth_first_release", "realtime_start": "truth_release_date"})[
        ["date", "truth_first_release", "truth_release_date"]
    ].set_index("date")


def build_episodes(gk: Gatekeeper, vintages: pd.DataFrame, start=DEFAULT_START, end=None,
                   series_id: str = TARGET) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (episodes, excluded). ``vintages`` is the same table the Gatekeeper holds."""
    truth = first_releases(vintages, series_id)
    last_month = truth.index.max() if end is None else pd.Timestamp(end)
    months = pd.date_range(pd.Timestamp(start), last_month, freq="MS")

    episodes, excluded = [], []
    for m in months:
        cutoff = m + pd.Timedelta(days=CUTOFF_DAY - 1)
        if m not in truth.index:
            excluded.append({"target_date": m.date(), "cutoff": cutoff.date(),
                             "reason": "no first release published"})
            continue
        t = truth.loc[m]
        if t["truth_release_date"] <= cutoff:  # would leak the answer
            excluded.append({"target_date": m.date(), "cutoff": cutoff.date(),
                             "reason": f"released {t['truth_release_date']:%Y-%m-%d}, on or before the cutoff"})
            continue
        seen = gk.latest(series_id, cutoff)
        episodes.append({
            "episode_id": f"{series_id}-{m:%Y-%m}",
            "episode": cutoff.date(),  # forecast date, as in the run log
            "target_month": f"{m:%Y-%m}",
            "target_first_release": float(t["truth_first_release"]),
            "last_known": seen["value"] if seen else None,
            "cutoff": cutoff.date(),
            "target_series": series_id,
            "target_date": m.date(),
            "last_visible_date": seen["date"].date() if seen else None,
            "last_visible_value": seen["value"] if seen else None,
            "gap_months": (m.year - seen["date"].year) * 12 + m.month - seen["date"].month if seen else None,
            "truth_first_release": float(t["truth_first_release"]),
            "truth_release_date": t["truth_release_date"].date(),
        })
    return pd.DataFrame(episodes), pd.DataFrame(excluded, columns=["target_date", "cutoff", "reason"])


def main(start=DEFAULT_START, end=None) -> pd.DataFrame:
    gk = load_gatekeeper(EVIDENCE_PATH)
    vintages = pd.read_parquet(EVIDENCE_PATH)
    episodes, excluded = build_episodes(gk, vintages, start, end)
    EPISODES_PATH.parent.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(EPISODES_PATH, index=False)
    excluded.to_csv(EXCLUDED_PATH, index=False)
    print(f"{len(episodes)} episodes ({episodes['target_date'].min()} to {episodes['target_date'].max()}) "
          f"-> {EPISODES_PATH.relative_to(REPO_ROOT)}")
    for r in excluded.itertuples():
        print(f"excluded {r.target_date}: {r.reason}")
    return episodes


if __name__ == "__main__":
    main(*sys.argv[1:3])
