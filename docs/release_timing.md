# Release timing check

Owner: Sonali. Code: `src/evaluation/release_timing.py`.

## Why this matters

The Gatekeeper shows a value only if its ALFRED `realtime_start` is on or
before the forecast cutoff (the 20th of each month, 5 p.m. ET). That rule is
only as good as the `realtime_start` dates themselves, so this check confirms
they line up with the official release schedules and records which series are
not yet out by the 20th.

## Official release schedules

| Series | Publisher and release | Usual timing (ET) | Reference period |
| --- | --- | --- | --- |
| UNRATE | BLS, Employment Situation | First Friday of the month (sometimes the second), 8:30 a.m. | Previous month |
| PAYEMS | BLS, Employment Situation | Same release as UNRATE | Previous month |
| JTSJOL | BLS, Job Openings and Labor Turnover Survey (JOLTS) | Usually the first or second Tuesday, 10:00 a.m. | Two months earlier |
| INDPRO | Federal Reserve Board, G.17 Industrial Production and Capacity Utilization | Around the 15th to 17th, 9:15 a.m. | Previous month |
| ICSA | DOL, Unemployment Insurance Weekly Claims | Every Thursday (Wednesday in some holiday weeks), 8:30 a.m. | Week ending the previous Saturday |

All of these releases come out in the morning, before the 5 p.m. cutoff, so a
release dated on the 20th itself counts as available. This matches the
Gatekeeper's `<=` rule.

Official calendars used for manual checks:

- BLS release calendar: https://www.bls.gov/schedule/
- Federal Reserve G.17 release dates: https://www.federalreserve.gov/releases/g17/
- DOL weekly claims: https://www.dol.gov/ui/data.pdf

## Method

1. For each series, take the first release of every observation since
   January 2023 (the earliest `realtime_start` for that date in ALFRED).
2. Check each first release against the usual pattern in the table above
   (weekday, day-of-month window, and how many months after the reference
   month it came out). Releases that break the pattern are flagged and checked
   by hand against the official calendar. Nothing is changed automatically.
3. For every forecast date (the 20th, Jan 2024 to Sep 2026), record the newest
   observation of each series already published, and list the cases where the
   month a forecaster would normally expect was still unpublished on the 20th.

Run it with:

```bash
python -m src.evaluation.release_timing            # downloads ALFRED with your FRED key
python -m src.evaluation.release_timing --evidence data/processed/evidence_tables/vintages.parquet
```

CSVs go to `results/release_timing/`, and the block below is regenerated.

## Known exception

The October 2025 federal government shutdown delayed BLS releases. The
October 2025 unemployment rate was never published, and the September 2025
value came out on November 20, 2025 instead of early October. The Gatekeeper
returns "not available" rather than filling in a value, and October 2025 is
excluded from scoring.

## Results

<!-- BEGIN GENERATED RESULTS -->

_Not run yet. Run the command above to fill this section._

<!-- END GENERATED RESULTS -->

## Takeaways for the paper

_Fill in after running: which series are routinely one month staler on the
20th (JOLTS by design), whether any INDPRO or Employment Situation releases
slipped past the 20th, and confirmation that ALFRED dates match the official
calendars apart from the flagged cases._
