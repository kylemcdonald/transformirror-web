#!/bin/bash

cleanup() {
    kill -- -$$
    exit 1
}

trap cleanup SIGINT

(
    TOTAL_DEVICES=$(python3 -c "import torch; print(torch.cuda.device_count())")
    FINAL_DEVICE=$((TOTAL_DEVICES - 1))
    for i in $(seq 0 $FINAL_DEVICE); do
        echo "Starting worker $i"
        # export HF_HOME=/workspace/.cache
        # source venv/bin/activate
        CUDA_VISIBLE_DEVICES=$i /home/bizon/anaconda3/bin/python3 worker.py &
    done
    wait
) &

wait $!
