#!/usr/bin/env bash
# Serve Qwen2.5-3B-Instruct with vLLM's OpenAI-compatible API (foreground; stop with Ctrl+C).
#
#   bash scripts/start_vllm.sh
#
# Same model settings as the authors (README / run_k_sweep.sh): --dtype float16 --max-model-len 8192.
# Differences, none of which change what the model computes for a request:
#   - no --enforce-eager: the authors needed it on older GPUs; CUDA graphs only make decoding faster
#   - --gpu-memory-utilization 0.55 instead of 0.95: only how much VRAM vLLM reserves (weights 5.8 GB + KV cache)
#   - VLLM_USE_FLASHINFER_SAMPLER=0: FlashInfer's sampler JIT-compiles with nvcc, which a WSL install without the
#     CUDA toolkit lacks; vLLM's PyTorch sampler samples from the same distribution
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${PORT:-30000}"
GPU_UTIL="${GPU_UTIL:-0.55}"
PY="${PY:-$HOME/venvs/amem/bin/python}"

export VLLM_USE_FLASHINFER_SAMPLER=0
exec "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$PORT" --host 0.0.0.0 \
  --dtype float16 --max-model-len 8192 \
  --gpu-memory-utilization "$GPU_UTIL"
