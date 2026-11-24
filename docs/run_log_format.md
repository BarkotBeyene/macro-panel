# Run log format (for every forecasting system)

Owner: Sonali. Code: `src/evaluation/runlog.py`.

Every system (no-change, AR(1), Chronos-2, TimesFM-3, single agent, panel,
majority vote) saves one JSON line per forecast so the same scoring code reads
all of them. Save to `results/forecasts/<run_id>/runs.jsonl`, one folder per
run, never overwriting an earlier run. `runlog.write_runs(records, path)`
appends records for you.

## Fields

| Field | Required | Type | Example |
| --- | --- | --- | --- |
| `episode` | yes | forecast date | `"2024-03-20"` |
| `target_month` | yes | month being nowcast | `"2024-03"` |
| `system` | yes | system name | `"panel"`, `"ar1"`, `"chronos2"` |
| `forecast` | yes | point forecast of first-release UNRATE (%) | `3.9` |
| `lo80`, `hi80` | if available | 10th and 90th percentiles | `3.7`, `4.1` |
| `cited` | agents | evidence the Chair says drove the forecast | `[{"series_id": "PAYEMS", "direction": "down", "rank": 1}]` |
| `acquired` | agents | series requested from the Gatekeeper | `["PAYEMS", "ICSA"]` |
| `evidence` | panel, single agent | exactly what the Chair was shown, keyed by series | `{"PAYEMS": "...", "ICSA": "..."}` |
| `credits` | agents | evidence budget spent | `2` |
| `tokens`, `n_calls` | agents | LLM tokens and calls | `1840`, `5` |
| `runtime_s` | all | seconds per forecast | `11.2` |
| `model`, `prompt_version`, `seed`, `vintage` | all | provenance | `"llama3.1:8b"`, `"panel-v1"`, `0`, `"2024-03"` |

`direction` is which way that evidence pushed the unemployment forecast:
`"up"` raised it, `"down"` lowered it.

## What the faithfulness test needs from the panel (Barkot)

1. Save `evidence` and `cited` for every panel and single-agent forecast.
2. Expose one function the test can call to rerun only the Chair's final
   decision on modified evidence, everything else held fixed:

   ```python
   # src/agents/chair.py
   def redecide(record: dict, evidence: dict) -> float:
       """Rerun the Chair's final decision for this saved record on `evidence`; return the point forecast."""
   ```

Then: `python -m src.evaluation.faithfulness results/forecasts/<panel_run> --decide src.agents.chair:redecide --repeats 3 --out results/faithfulness/<run_id>`

## Episodes file for scoring (Vaishnavi)

`data/processed/forecast_episodes/episodes.csv` with at least `target_month`
and `target_first_release`; add `last_known` (the latest UNRATE visible on the
forecast date) to get directional accuracy.

## Scoring commands

```bash
python -m src.evaluation.forecasting results/forecasts/<run1> results/forecasts/<run2> \
    --truth data/processed/forecast_episodes/episodes.csv --reference no_change --out results/scores/<run_id>
python -m src.evaluation.costs results/forecasts/<run1> results/forecasts/<run2>
```
