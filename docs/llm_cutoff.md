# LLM training cutoff and memorization probe

Owner: Sonali. Code: `src/evaluation/memorization.py`.

## Documented cutoff

| Model | Hugging Face ID | Documented cutoff | Source |
| --- | --- | --- | --- |
| Llama 3.1 8B Instruct | `meta-llama/Llama-3.1-8B-Instruct` | December 2023 | Meta's Llama 3.1 model card ("Data Freshness: The pretraining data has a cutoff of December 2023"; the training data table lists the knowledge cutoff as December 2023) |

Model release date: July 23, 2024.

## What this means for our evaluation

- Our forecast dates run from January 2024 to September 2026, so every target
  month in the main window was first published after the cutoff. That gives
  about 32 scorable months (October 2025 is excluded because it was never
  published).
- The December 2023 unemployment rate was first published on January 5, 2024,
  after the cutoff, so the probe counts it as post-cutoff. The probe labels
  months by their first release date, not by the month they describe.
- A documented cutoff is not proof the model saw nothing later. Meta does not
  list every source, and post-training data could be newer. The probe below
  tests this directly instead of assuming it.

## Memorization probe (H2)

We ask the model, with no evidence, for the first-reported unemployment rate
of each month from January 2022 to September 2026, and score its answer
against the ALFRED first release.

- Prompt version: `recall-v1` (stored with each run in `run.json`)
- Decoding: temperature 0, one answer per month by default
- Naive comparator: the last value published before the cutoff (3.7% for
  November 2023). A model with no post-cutoff knowledge should do about as
  well as this guess after the cutoff.

What we expect if the cutoff holds:

| Period | Model error | Compared with the naive guess |
| --- | --- | --- |
| Pre-cutoff (2022 to Nov 2023) | Low, many exact matches | Much better |
| Post-cutoff (Dec 2023 to Sep 2026) | Higher | About the same, no better |

If post-cutoff accuracy is clearly better than the naive guess, the model may
have seen later data, and we report it as a contamination risk.

Run it with:

```bash
# Ollama (easiest on a laptop)
ollama pull llama3.1:8b
python -m src.evaluation.memorization --backend openai --model llama3.1:8b

# vLLM server (on a GPU)
python -m src.evaluation.memorization --backend openai --model meta-llama/Llama-3.1-8B-Instruct --base-url http://localhost:8000/v1

# transformers directly, 4-bit to fit a smaller GPU (needs HF_TOKEN and bitsandbytes)
python -m src.evaluation.memorization --backend hf --model meta-llama/Llama-3.1-8B-Instruct --load-4bit

# quick smoke test on 3 months per period
python -m src.evaluation.memorization --limit 3
```

Results go to `results/memorization/<run_id>/` (`probe.csv`, `summary.csv`,
`run.json`).

## Results

_Fill in from `summary.csv` after the run: MAE, exact-match rate and naive
MAE for pre-cutoff and post-cutoff months, plus one or two example replies._

| Period | Months | Model MAE | Exact match | Naive MAE |
| --- | --- | --- | --- | --- |
| Pre-cutoff | | | | |
| Post-cutoff | | | | |
