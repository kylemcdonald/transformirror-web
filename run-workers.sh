#!/bin/bash

cleanup() {
    kill -- -$$
    exit 1
}

trap cleanup SIGINT

(
    for i in {0..7}; do
        echo "Starting worker $i"
        CUDA_VISIBLE_DEVICES=$i python3 worker.py &
    done
    wait
) &

wait $!
