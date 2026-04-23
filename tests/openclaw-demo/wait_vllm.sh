#!/usr/bin/env bash
# Block until both vLLM endpoints respond to /v1/models (timeout 15 min each).
set -u
for port in 8088 8089; do
  echo "waiting for http://127.0.0.1:${port}/v1/models ..."
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "  :${port} is up"
      break
    fi
    sleep 5
  done
done
