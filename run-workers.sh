#!/bin/bash

cleanup() {
    kill -- -$$
    exit 1
}

trap cleanup SIGINT

(
    source .venv/bin/activate
    TOTAL_DEVICES=$(python -c "import torch; print(torch.cuda.device_count())")
    FINAL_DEVICE=$((TOTAL_DEVICES - 1))
    for i in $(seq 0 $FINAL_DEVICE); do
        echo "Starting worker $i"
        export HF_HOME=/workspace/.cache
        CUDA_VISIBLE_DEVICES=$i python worker.py &
    done
    wait
) &

wait $!
