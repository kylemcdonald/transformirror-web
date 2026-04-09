#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

cleanup() {
    kill -- -$$
    exit 1
}

trap cleanup SIGINT

export HF_HOME="${HF_HOME:-/workspace/.cache}"
export PYTHONUNBUFFERED=1

(
    source .venv/bin/activate
    TOTAL_DEVICES=$(python -c "import torch; print(torch.cuda.device_count())")
    FINAL_DEVICE=$((TOTAL_DEVICES - 1))
    for i in $(seq 0 $FINAL_DEVICE); do
        echo "Starting worker $i"
        CUDA_VISIBLE_DEVICES=$i python worker.py &
    done
    wait
) &

wait $!
