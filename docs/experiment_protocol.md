# Experiment protocol

Sections marked TBD are team decisions that have not been made yet.

## Target and forecast episodes

- Target series: U.S. unemployment rate (ALFRED vintages).
- Forecast dates: TBD
- Horizons: TBD

## Evidence list

Indicators the panel may request through the Gatekeeper: TBD

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
