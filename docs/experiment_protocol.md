# Experiment protocol

Sections marked TBD are team decisions that have not been made yet.

## Target and forecast episodes

- Target series: U.S. unemployment rate, `UNRATE` (ALFRED vintages).
- Forecast dates: the 20th of each month, 5 p.m. ET. A release dated on the
  cutoff day counts as visible (releases come out in the morning).
- Horizon: nowcast of the current month (target = the month of the cutoff),
  which BLS first publishes in early the next month.
- Truth: the first official release of the target month (ALFRED), not later
  revisions.
- Window: January 2024 onward (after Llama 3.1's December 2023 training
  cutoff) through the latest month with a first release. Months never
  published (October 2025, government shutdown) are excluded and listed in
  `data/processed/forecast_episodes/excluded.csv`.
- Built by `python -m src.build_episodes` into
  `data/processed/forecast_episodes/episodes.csv`.

## Evidence list

Indicators the panel may request through the Gatekeeper (all releases and
revisions, built by `python -m src.build_evidence`; descriptions in
`src/build_evidence.py`, `SERIES`):

| Series | Indicator | Frequency |
| --- | --- | --- |
| UNRATE | Unemployment rate (target; its own history is visible) | monthly |
| PAYEMS | Nonfarm payroll employment | monthly |
| ICSA | Initial unemployment claims | weekly |
| CCSA | Continued unemployment claims | weekly |
| JTSJOL | Job openings (JOLTS) | monthly |
| INDPRO | Industrial production | monthly |
| CPIAUCSL | Consumer prices (CPI) | monthly |
| FEDFUNDS | Effective federal funds rate | monthly |

## Models

| Model | Role | Version |
| --- | --- | --- |
| AR(1) | Statistical baseline | statsmodels 0.15.0 |
| Chronos-2 | Foundation-model baseline | chronos-forecasting 2.3.2 |
| TimesFM-3 | Foundation-model baseline | timesfm 3.0.2, `google/timesfm-3.0-pytorch` |

Primary comparison: Chronos-2 and TimesFM-3 see the same unemployment history,
forecast dates and horizons. TimesFM-3 gets no extra covariates here; a
multivariate run is an extension only.

Every baseline returns: forecast value, 80% prediction interval (10th and 90th
percentiles), forecast origin, model version, evidence consumed.

## Agent systems (Milestone II pilot)

Code: `src/agents/` (run with `bash scripts/run_milestone2.sh`).

| Setting | Value |
| --- | --- |
| LLM | Llama 3.1 8B Instruct (`llama3.1:8b` via Ollama, same model as the memorization probe) |
| Temperature | 0.3 for every role; seed 0 for the run, no seed when the faithfulness test reruns the Chair |
| Max new tokens | Analyst 300, Economist 350, Skeptic 300, Chair step 150, Chair decision 400 |
| Evidence budget | 4 credits per forecast; every menu item costs 1; UNRATE history is free |
| Menu | PAYEMS, ICSA, CCSA, JTSJOL, INDPRO, CPIAUCSL, FEDFUNDS, Chronos-2 nowcast, TimesFM-3 nowcast |
| Rounds | at most 3 (Analyst, Economist, Skeptic, then the Chair buys evidence or commits) |
| Final decision | separate Chair call: forecast, 80% interval, cited items with direction and rank, explanation |
| Single agent | same model, menu, budget and final-decision prompt, no discussion |
| Majority vote | 3 independent single agents (seeds 0, 1, 2); median forecast and interval; items cited by at least 2 |
| Pilot episodes | 12 of the 33, evenly spaced (first and last included) |
| Faithfulness | remove each evidence item, and all labor-market series together; 3 repeats; moved if the shift exceeds max(0.05 pp, 2 x repeat SD) |

## Stages

1. Controlled foundation-model comparison (above).
2. Panel: sees both models' forecasts, requests further evidence, deliberates.
3. Faithfulness tests on the saved panel record: evidence removal and
   counterfactuals, cited versus uncited evidence. The panel cannot revise its
   explanation after the fact.

## Logged per run

- Model identifiers and versions
- Forecast date and data vintage
- Parameters and prompt version
- Initial forecast, requested evidence, updated forecasts
- Final explanation and cited indicators
- Costs
- Results
