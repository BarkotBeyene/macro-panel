# macro-panel

An evidence-seeking AI economist panel for macroeconomic forecasting, built to
test whether the panel's explanations are faithful to the evidence that
actually drives its forecasts.

**Research question:** Can an evidence-seeking AI economist panel produce
faithful explanations of macroeconomic forecasts across different time-series
foundation models (Chronos-2 and TimesFM-3), and do those explanations remain
reliable when the underlying forecasts fail?

## Setup

Python 3.14.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
cp .env.example .env   # then add your FRED API key
```

## Layout

| Path | Purpose |
| --- | --- |
| `src/gatekeeper.py` | Enforces historical availability; agents never bypass it |
| `src/baselines/` | AR(1), Chronos-2 and TimesFM-3 behind one common interface |
| `src/agents/` | Analyst, Economist, Skeptic, Chair and the orchestrator |
| `src/evaluation/` | Forecast accuracy, faithfulness, memorization, costs |
| `data/raw/` | ALFRED vintages, tracked in `manifest.csv` |
| `data/processed/` | Forecast episodes and evidence tables |
| `results/` | One subfolder per run; never overwrite earlier runs |
| `docs/` | Experiment protocol |

## Planned files

The package folders exist now. Each file below is added in its owner's PR,
together with real code; do not create empty placeholders.

```
src/
├── gatekeeper.py            (exists, stub)
├── baselines/
│   ├── interface.py         common forecast result: value, interval, origin, model version, evidence consumed
│   ├── ar1.py
│   ├── chronos2.py
│   └── timesfm3.py
├── agents/
│   ├── analyst.py
│   ├── economist.py
│   ├── skeptic.py
│   ├── chair.py
│   └── orchestrator.py
└── evaluation/
    ├── forecasting.py       accuracy, interval coverage and calibration
    ├── faithfulness.py      evidence-removal and counterfactual tests (H1)
    ├── memorization.py      historical knowledge leakage (H2)
    └── costs.py
```

## Branches and pull requests

- `main` is protected: no direct commits after the initial one.
- Work on a branch, open a PR into `main`, and get one teammate's review before merging.
- One PR per planned file or small group of related files, with its tests.
- Never commit `.env`, raw API keys or anything under `.venv/`.

## Research rules

1. The Gatekeeper is independent of the agents.
2. The panel's outputs are saved before any faithfulness test runs.
3. Every run records models, forecast date, data vintage, prompt version, evidence and costs.
