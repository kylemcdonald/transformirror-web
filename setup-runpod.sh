#!/bin/bash
# Set up and start transformirror on a RunPod pod using the
# runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04 image.
set -euo pipefail

cd "$(dirname "$0")"

export HF_HOME="${HF_HOME:-/workspace/.cache}"
mkdir -p "$HF_HOME"

apt-get update
apt-get install -y libturbojpeg tmux

TORCH_INDEX=https://download.pytorch.org/whl/cu121
STABLE_FAST_WHEEL=https://github.com/chengzeyi/stable-fast/releases/download/v0.0.13.post3/stable_fast-0.0.13.post3+torch210cu121-cp310-cp310-manylinux2014_x86_64.whl

python3.10 -m venv .venv
.venv/bin/pip install --upgrade pip wheel setuptools
.venv/bin/pip install --extra-index-url "$TORCH_INDEX" \
    'torch==2.1.0+cu121' \
    'torchvision==0.16.0+cu121' \
    'xformers==0.0.22.post7' \
    "stable_fast @ $STABLE_FAST_WHEEL"
.venv/bin/pip install --extra-index-url "$TORCH_INDEX" -r requirements.txt \
    'torch==2.1.0+cu121' \
    'xformers==0.0.22.post7'

.venv/bin/python download_files.py

./install-workers.sh
./install-server.sh
