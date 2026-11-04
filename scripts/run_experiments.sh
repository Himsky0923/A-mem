#!/usr/bin/env bash
# Experiments for the A-Mem reproduction (LoCoMo, robust pipeline). Run from the repository root, one step at a time:
#
#   bash scripts/run_experiments.sh ksweep      # Qwen, evolution on, seed 1: k = 10 20 30 40 50 (pick K)
#   K=10 bash scripts/run_experiments.sh qwen   # Qwen: evolution on / off x seeds 1 2 3
#   K=10 bash scripts/run_experiments.sh deepseek   # DeepSeek (stands in for GPT-4o-mini): same six runs
#   K=10 bash scripts/run_experiments.sh full   # optional: Qwen, evolution on, seed 1, whole dataset
#   bash scripts/run_experiments.sh aggregate   # results/summary.md + CSVs
#
# Qwen steps need the vLLM server (bash scripts/start_vllm.sh in another terminal). DeepSeek needs OPENAI_API_KEY
# (in .env or the environment). Each run writes results/<step>/<name>.json and logs/run_<step>_<name>.out.
#
# Memories are cached per backend, model, evolution setting, seed and run tag (cached_memories_robust/). The first
# k of the sweep builds the seed-1 memories that the other k values and the seed-1 baseline of 'qwen' reuse.
#
# Variables: K (retrieve_k for qwen/deepseek/full, default 10), SEEDS (default "1 2 3"), RATIO (default 0.3:
# the first 3 of 10 conversations, 497 questions), PARALLEL (runs at the same time, default 1), PY (python).
set -euo pipefail

STEP="${1:-}"
PY="${PY:-$HOME/venvs/amem/bin/python}"
K="${K:-10}"
SEEDS="${SEEDS:-1 2 3}"
RATIO="${RATIO:-0.3}"
PARALLEL="${PARALLEL:-1}"
PORT="${PORT:-30000}"
QWEN="Qwen/Qwen2.5-3B-Instruct"
SWEEP_K="${SWEEP_K:-10 20 30 40 50}"

cd "$(dirname "$0")/.."
mkdir -p logs results

FAILED=0

wait_for_slot() {  # at most PARALLEL runs at a time
  while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do
    wait -n || FAILED=$((FAILED + 1))
  done
}

finish() {
  while [ "$(jobs -rp | wc -l)" -gt 0 ]; do
    wait -n || FAILED=$((FAILED + 1))
  done
  if [ "$FAILED" -gt 0 ]; then
    echo "$FAILED run(s) failed; see logs/run_*.out"
    exit 1
  fi
  echo "Step '$STEP' done."
}

# run <step> <name> <args...>: one evaluation in the background, output to results/<step>/<name>.json
run() {
  local step=$1 name=$2
  shift 2
  mkdir -p "results/$step"
  wait_for_slot
  echo "[$(date '+%F %T')] start $step/$name"
  "$PY" test_advanced_robust.py --dataset data/locomo10.json --ratio "$RATIO" \
    --output "results/$step/$name.json" "$@" > "logs/run_${step}_${name}.out" 2>&1 &
}

need_vllm() {
  if ! curl -sf "localhost:$PORT/v1/models" > /dev/null; then
    echo "No vLLM server on port $PORT. Start it first: bash scripts/start_vllm.sh"
    exit 1
  fi
}

qwen_args=(--backend vllm --model "$QWEN" --sglang_port "$PORT")   # --sglang_port is also the vLLM port
deepseek_args=(--backend openai --model deepseek-chat --base_url https://api.deepseek.com)

case "$STEP" in
  ksweep)
    need_vllm
    # the first k builds the memories; the others only answer from them, so they start after it
    read -r -a ks <<< "$SWEEP_K"
    run ksweep "qwen_k${ks[0]}" "${qwen_args[@]}" --seed 1 --run_tag evo_s1 --retrieve_k "${ks[0]}"
    wait "$!" || FAILED=$((FAILED + 1))
    for k in "${ks[@]:1}"; do
      run ksweep "qwen_k${k}" "${qwen_args[@]}" --seed 1 --run_tag evo_s1 --retrieve_k "$k"
    done
    finish
    "$PY" scripts/aggregate.py results/ksweep/*.json --out results/ksweep/summary > /dev/null
    echo "k-sweep table: results/ksweep/summary.md (choose K by overall F1)"
    ;;
  qwen|deepseek)
    if [ "$STEP" = qwen ]; then need_vllm; args=("${qwen_args[@]}"); else args=("${deepseek_args[@]}"); fi
    for seed in $SEEDS; do
      run main "${STEP}_evo_k${K}_s${seed}" "${args[@]}" --seed "$seed" --run_tag "evo_s${seed}" --retrieve_k "$K"
      run main "${STEP}_noevo_k${K}_s${seed}" "${args[@]}" --seed "$seed" --run_tag "noevo_s${seed}" --retrieve_k "$K" \
        --no_evolution
    done
    finish
    ;;
  full)
    need_vllm
    RATIO=1.0 run full "qwen_evo_k${K}_s1" "${qwen_args[@]}" --seed 1 --run_tag evo_s1 --retrieve_k "$K"
    finish
    ;;
  aggregate)
    "$PY" scripts/aggregate.py results/main/*.json --out results/summary
    ;;
  *)
    sed -n '2,19p' "$0"
    exit 1
    ;;
esac
