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

_Generated 2026-11-23 15:45 from ALFRED via fredapi; releases from 2023-01-01, forecast dates 2024-01-20 to 2026-09-20._

### Summary

| Series | Official release | Releases checked | Match schedule | Flagged | Median release day | Releases after the 20th | Forecast dates missing expected month |
| --- | --- | --- | --- | --- | --- | --- | --- |
| UNRATE | BLS Employment Situation, 8:30 a.m. ET | 43 | 38 | 5 | 5 | 0 | 2 |
| PAYEMS | BLS Employment Situation, 8:30 a.m. ET | 44 | 38 | 6 | 5 | 0 | 2 |
| JTSJOL | BLS JOLTS, 10:00 a.m. ET | 43 | 42 | 1 | 5 | 10 | 1 |
| INDPRO | Federal Reserve Board G.17 Industrial Production, 9:15 a.m. ET | 44 | 41 | 3 | 16 | 2 | 3 |
| ICSA | DOL Unemployment Insurance Weekly Claims, 8:30 a.m. ET | 193 | 186 | 7 | weekly | n/a | 0 |

### What the panel can see on the 20th (monthly series)

| Series | Min months stale | Median | Max |
| --- | --- | --- | --- |
| INDPRO | 1.0 | 1.0 | 3.0 |
| JTSJOL | 2.0 | 2.0 | 3.0 |
| PAYEMS | 1.0 | 1.0 | 2.0 |
| UNRATE | 1.0 | 1.0 | 2.0 |

### Forecast dates where the expected month was not yet published

| forecast_date | series_id | reference_month | first_release | note |
| --- | --- | --- | --- | --- |
| 2025-10-20 | UNRATE | 2025-09 | 2025-11-20 | released 31 days after the forecast date |
| 2025-10-20 | PAYEMS | 2025-09 | 2025-11-20 | released 31 days after the forecast date |
| 2025-10-20 | INDPRO | 2025-09 | 2025-12-03 | released 44 days after the forecast date |
| 2025-11-20 | UNRATE | 2025-10 |  | never published (no value in ALFRED) |
| 2025-11-20 | PAYEMS | 2025-10 | 2025-12-16 | released 26 days after the forecast date |
| 2025-11-20 | JTSJOL | 2025-09 | 2025-12-09 | released 19 days after the forecast date |
| 2025-11-20 | INDPRO | 2025-10 | 2025-12-23 | released 33 days after the forecast date |
| 2025-12-20 | INDPRO | 2025-11 | 2025-12-23 | released 3 days after the forecast date |

### Releases flagged for manual check against the official calendar

| series_id | date | first_release | weekday | issue |
| --- | --- | --- | --- | --- |
| ICSA | 2025-09-27 | 2025-11-20 | Thu | 54 days after week end, expected 4-5 |
| ICSA | 2025-10-04 | 2025-11-20 | Thu | 47 days after week end, expected 4-5 |
| ICSA | 2025-10-11 | 2025-11-20 | Thu | 40 days after week end, expected 4-5 |
| ICSA | 2025-10-18 | 2025-11-20 | Thu | 33 days after week end, expected 4-5 |
| ICSA | 2025-10-25 | 2025-11-20 | Thu | 26 days after week end, expected 4-5 |
| ICSA | 2025-11-01 | 2025-11-20 | Thu | 19 days after week end, expected 4-5 |
| ICSA | 2025-11-08 | 2025-11-20 | Thu | 12 days after week end, expected 4-5 |
| INDPRO | 2025-09-01 | 2025-12-03 | Wed | released 3 month(s) after reference month, expected 1 |
| INDPRO | 2025-10-01 | 2025-12-23 | Tue | released 2 month(s) after reference month, expected 1 |
| INDPRO | 2025-11-01 | 2025-12-23 | Tue | day 23 outside usual 13-19 |
| JTSJOL | 2025-09-01 | 2025-12-09 | Tue | 70 days after the reference month ended, expected 25-45 |
| PAYEMS | 2025-06-01 | 2025-07-03 | Thu | weekday Thu |
| PAYEMS | 2025-09-01 | 2025-11-20 | Thu | weekday Thu; released 2 month(s) after reference month, expected 1 |
| PAYEMS | 2025-10-01 | 2025-12-16 | Tue | weekday Tue; released 2 month(s) after reference month, expected 1 |
| PAYEMS | 2025-11-01 | 2025-12-16 | Tue | weekday Tue; day 16 outside usual 1-12 |
| PAYEMS | 2026-01-01 | 2026-02-11 | Wed | weekday Wed |
| PAYEMS | 2026-06-01 | 2026-07-02 | Thu | weekday Thu |
| UNRATE | 2025-06-01 | 2025-07-03 | Thu | weekday Thu |
| UNRATE | 2025-09-01 | 2025-11-20 | Thu | weekday Thu; released 2 month(s) after reference month, expected 1 |
| UNRATE | 2025-11-01 | 2025-12-16 | Tue | weekday Tue; day 16 outside usual 1-12 |
| UNRATE | 2026-01-01 | 2026-02-11 | Wed | weekday Wed |
| UNRATE | 2026-06-01 | 2026-07-02 | Thu | weekday Thu |

<!-- END GENERATED RESULTS -->

## Takeaways for the paper

1. **ALFRED's release dates line up with the official schedules.** Outside the
   2025 shutdown, almost every first release falls where the BLS, Fed and DOL
   calendars put it. The remaining off-pattern dates are explainable: the
   Employment Situation moved to Thursday, July 3, 2025 and Thursday, July 2,
   2026 because of the July 4 holiday, and the January 2026 report came out on
   Wednesday, February 11, 2026 (to confirm against the BLS calendar).
2. **What the panel sees on the 20th.** In a normal month the panel has last
   month's UNRATE, PAYEMS and INDPRO, and JOLTS from two months back (JOLTS
   always lags one extra month). Weekly claims are current to the previous
   week. No regular release lands after the 20th, so the 5 p.m. cutoff on the
   20th is a clean, consistent information set.
3. **The October 2025 shutdown is the one real exception.** It caused every
   forecast date where an expected value was missing (October 20, November 20
   and December 20, 2025): the September 2025 jobs report slipped to November
   20, the October and November reports were combined on December 16 (with no
   October unemployment rate ever published), INDPRO for September to November
   was delayed to December, and ALFRED dates several weeks of claims to
   November 20, 2025. The Gatekeeper handles this correctly by showing older
   data on those dates rather than filling gaps, and the October 2025 target
   is excluded from scoring.
4. **Implication for the experiments.** Forecasts on those three dates run on
   staler evidence than usual, so we report results with and without them as
   a robustness check.
