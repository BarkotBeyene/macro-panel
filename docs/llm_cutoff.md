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

- Prompt version: `recall-v2` (stored with each run in `run.json`)
- If the model declines (no number in its reply), we ask once more for its
  single best guess. The decline itself is recorded, and the decline rate is
  reported per period, since refusing after the cutoff is evidence the model
  knows where its knowledge stops.
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

Run `20261123-171345-llama3.1-8b` (Llama 3.1 8B via Ollama, prompt `recall-v2`,
temperature 0, one answer per month, January 2022 to September 2026, October
2025 excluded because it was never published).

| Period | Months | Decline rate | Model MAE | Exact match | Within 0.2 pts | Naive MAE | Naive exact match |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pre-cutoff (Jan 2022 to Nov 2023) | 23 | 0% | 0.10 | 61% | 83% | 0.13 | 22% |
| Post-cutoff (Dec 2023 to Sep 2026) | 33 | 100% | 0.44 | 15% | 45% | 0.45 | 6% |

The naive guess is 3.7%, the November 2023 value, which was the last one
published before the cutoff.

### Takeaways

1. **The model remembers pre-cutoff unemployment.** With no evidence at all it
   named the exact first-release rate for 14 of 23 months in 2022 and 2023,
   almost three times as often as the naive guess (61% vs 22%). Its misses
   cluster in late 2023 (September to November), the months closest to the
   cutoff and least covered in training data.
2. **It does not know post-cutoff values.** It declined every month from
   December 2023 on ("I don't have the information"). When pushed for a
   guess, it repeated generic values (4.5% for most of early 2024) and its
   error was the same as the naive guess (0.44 vs 0.45). There is no sign it
   saw 2024 to 2026 unemployment data.
3. **Implication.** Our January 2024 to September 2026 evaluation window is
   clean for this model, so the panel has to rely on the evidence it requests
   rather than recall. Pre-cutoff months, where the model could answer from
   memory, are the right comparison period for the H2 faithfulness test.
4. **Caveat.** This is one prompt at temperature 0. A model can still hold
   partial knowledge that a direct question does not surface, so H2 compares
   periods rather than claiming any period is contamination free.
