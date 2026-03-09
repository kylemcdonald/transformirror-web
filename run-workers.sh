#!/bin/bash

PYTHON_BIN=/home/bizon/anaconda3/envs/transformirror-py31020-clean/bin/python
export PYTHONNOUSERSITE=1

cleanup() {
    kill -- -$$
    exit 1
}

trap cleanup SIGINT

(
    TOTAL_DEVICES=$($PYTHON_BIN -c "import torch; print(torch.cuda.device_count())")
    FINAL_DEVICE=$((TOTAL_DEVICES - 1))
    for i in $(seq 0 $FINAL_DEVICE); do
        echo "Starting worker $i"
        # export HF_HOME=/workspace/.cache
        # source venv/bin/activate
        CUDA_VISIBLE_DEVICES=$i $PYTHON_BIN worker.py &
    done
    wait
) &

wait $!
