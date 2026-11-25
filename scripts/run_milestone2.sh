#!/usr/bin/env bash
# Milestone II pilot on the GPU server: agents -> faithfulness -> report.
#
# Needs an Ollama server already running with llama3.1:8b, e.g.
#     OLLAMA_NUM_PARALLEL=4 ollama serve
# Run inside tmux so a dropped SSH connection does not stop it:
#     bash scripts/run_milestone2.sh                                # new pilot run
#     bash scripts/run_milestone2.sh results/forecasts/<run_id>     # resume a run
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate

export MACRO_LLM_BACKEND="${MACRO_LLM_BACKEND:-openai}"
export MACRO_LLM_MODEL="${MACRO_LLM_MODEL:-llama3.1:8b}"
export MACRO_LLM_URL="${MACRO_LLM_URL:-http://localhost:11434/v1}"
WORKERS="${WORKERS:-4}"
REPEATS="${REPEATS:-3}"
mkdir -p logs

echo "== 1/3 agents (panel, single agent, majority vote) on the 12 pilot episodes"
if [ -n "${1:-}" ]; then
  RUN="$1"
  python -m src.agents.orchestrator --resume "$RUN" --workers "$WORKERS" 2>&1 | tee -a "logs/$(basename "$RUN").log"
else
  python -m src.agents.orchestrator --systems panel single_agent majority_vote --episodes pilot \
      --workers "$WORKERS" 2>&1 | tee logs/agents_latest.log
  RUN="$(ls -d results/forecasts/*-agents | tail -1)"
  mv logs/agents_latest.log "logs/$(basename "$RUN").log"
fi
ID="$(basename "$RUN")"
echo "agent run: $RUN"

echo "== 2/3 faithfulness (evidence removal, $REPEATS repeats), panel and single agent in parallel"
for SYS in panel single_agent; do
  OUT="results/faithfulness/${ID}-${SYS}"
  if [ -f "$OUT/summary.csv" ]; then echo "  $OUT exists, skipping"; continue; fi
  rm -rf "$OUT"
  python -m src.evaluation.faithfulness "$RUN" --decide src.agents.chair:redecide --systems "$SYS" \
      --repeats "$REPEATS" --out "$OUT" > "logs/${ID}-faithfulness-${SYS}.log" 2>&1 &
done
wait
tail -n 5 logs/${ID}-faithfulness-*.log

echo "== 3/3 report"
python -m src.agents.report "$RUN" --faithfulness "results/faithfulness/${ID}-panel" \
    "results/faithfulness/${ID}-single_agent"
echo "done: results/milestone2/${ID}/summary.md"
