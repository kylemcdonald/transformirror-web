#!/bin/bash

cleanup() {
    kill -- -$$
    exit 1
}

trap cleanup SIGINT

TOTAL_DEVICES=$(python3 -c "import torch; print(torch.cuda.device_count())")
FINAL_DEVICE=$((TOTAL_DEVICES - 1))

for i in $(seq 0 $FINAL_DEVICE); do
    echo "Starting worker $i"
    CUDA_VISIBLE_DEVICES=$i $HOME/anaconda3/envs/transformirror/bin/python worker.py &
done

wait
