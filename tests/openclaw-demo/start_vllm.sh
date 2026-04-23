#!/usr/bin/env bash
# Start XtraGPT-7B on :8088 (GPU 0) and Qwen2.5-7B-Instruct on :8089 (GPU 1).
# Both in background, logs under logs/.

set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
mkdir -p logs

source /home/nuochen/miniconda3/etc/profile.d/conda.sh
conda activate llamafactory

start_one() {
  local name="$1" model="$2" port="$3" gpu="$4"
  if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
    echo "[$name] already up on :${port}"
    return 0
  fi
  echo "[$name] starting on GPU ${gpu}, port ${port}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup \
    vllm serve "$model" \
      --host 127.0.0.1 --port "$port" \
      --served-model-name "$model" \
      --max-model-len 16384 \
      --gpu-memory-utilization 0.85 \
      --disable-log-requests \
      > "logs/vllm_${name}.log" 2>&1 &
  echo "[$name] pid=$!"
}

start_one xtragpt  Xtra-Computing/XtraGPT-7B   8088 0
start_one qwen     Qwen/Qwen2.5-7B-Instruct    8089 1

echo
echo "tail -f logs/vllm_xtragpt.log logs/vllm_qwen.log"
