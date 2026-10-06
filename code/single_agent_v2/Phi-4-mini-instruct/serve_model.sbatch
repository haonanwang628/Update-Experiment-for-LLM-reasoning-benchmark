#!/bin/bash
#SBATCH --job-name=serve-model
#SBATCH --partition=general
#SBATCH --nodelist=<node>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.log

# Usage: sbatch --export=MODEL=<hf_repo>,PORT=<port> serve_model.sbatch
set -euo pipefail
export TMPDIR=~/tmp
mkdir -p "$TMPDIR"
# The compute node's nvcc is older than what flashinfer's JIT-compiled sampler
# kernels expect (fails on --compress-mode=size); use vLLM's built-in sampler instead.
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$MODEL" \
  --port "$PORT" \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --dtype bfloat16
