#!/bin/bash

cleanup() {
    kill -- -$$
    exit 1
}

trap cleanup SIGINT

(
    for i in {0..3}; do
        echo "Starting worker $i"
        export HF_HOME=/workspace/.cache
        source venv/bin/activate
        CUDA_VISIBLE_DEVICES=$i python3 worker.py &
    done
    wait
) &

wait $!
