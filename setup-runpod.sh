#!/bin/bash
# Set up and start transformirror on a RunPod pod using the
# runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04 image.
#
# If CONTROL_HOOK_URL and CONTROL_HOOK_TOKEN are set (as they are for pods
# launched by transformirror-control), each stage is also reported there.
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

export HF_HOME="${HF_HOME:-/workspace/.cache}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$HF_HOME"

report() {
    local stage=$1 detail=${2:-}
    echo "[setup] $(date -u +%H:%M:%S) $stage${detail:+: $detail}"
    if [ -n "${CONTROL_HOOK_URL:-}" ] && [ -n "${CONTROL_HOOK_TOKEN:-}" ]; then
        local body
        body=$(python3 -c 'import json, sys; print(json.dumps({"stage": sys.argv[1], "detail": sys.argv[2]}))' "$stage" "$detail")
        curl -fsS -m 10 -X POST "$CONTROL_HOOK_URL" \
            -H "Authorization: Bearer $CONTROL_HOOK_TOKEN" \
            -H "Content-Type: application/json" \
            --data "$body" >/dev/null 2>&1 || true
    fi
}

fail() {
    report failed "$1"
    exit 1
}

trap 'fail "setup-runpod.sh failed at line $LINENO"' ERR

GPU_COUNT=$(nvidia-smi -L | wc -l)
# -i 0 rather than `| head -n 1`: some drivers get SIGPIPE when head exits, which fails under pipefail.
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)

report system "Installing system packages"
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libturbojpeg tmux >/dev/null

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi

# Model weights download in parallel with the Python install.
report python "Installing Python packages, downloading models in parallel"
(
    uvx --quiet --from 'huggingface_hub[hf_transfer]==0.25.2' huggingface-cli download \
        stabilityai/sdxl-turbo \
        --include model_index.json '*/config.json' '*/*.fp16.safetensors' 'tokenizer*/*' 'scheduler/*' \
        >/dev/null
    uvx --quiet --from 'huggingface_hub[hf_transfer]==0.25.2' huggingface-cli download \
        madebyollin/taesdxl \
        --include config.json diffusion_pytorch_model.safetensors \
        >/dev/null
) > logs/models.log 2>&1 &
MODELS_PID=$!

uv venv --quiet --allow-existing --python /usr/bin/python3.10 .venv
uv pip install --quiet --python .venv/bin/python --index-strategy unsafe-best-match \
    -r requirements-runpod.txt

report models "Finishing model download"
if ! wait "$MODELS_PID"; then
    fail "Model download failed: $(tail -n 5 logs/models.log)"
fi
.venv/bin/python download_files.py >/dev/null 2>&1

report workers "Starting $GPU_COUNT worker(s) on $GPU_NAME"
touch logs/workers.log
WORKER_LOG_START=$(wc -l < logs/workers.log)
./install-workers.sh
./install-server.sh

worker_log() {
    tail -n +"$((WORKER_LOG_START + 1))" logs/workers.log
}

# Stable-fast compiles on the first warmup run, which takes a minute or two.
deadline=$((SECONDS + 900))
last=""
while true; do
    loaded=$(worker_log | grep -c ": model loaded" || true)
    warmed=$(worker_log | grep -c ": warmup finished" || true)
    if [ "$warmed" -ge "$GPU_COUNT" ]; then
        break
    fi
    if [ "$loaded" -lt "$GPU_COUNT" ]; then
        status="Loading model onto GPUs ($loaded/$GPU_COUNT)"
    else
        status="Compiling and warming up GPUs ($warmed/$GPU_COUNT ready)"
    fi
    if [ "$status" != "$last" ]; then
        report warmup "$status"
        last=$status
    fi
    if ! tmux has-session -t transformirror-workers 2>/dev/null; then
        fail "Workers exited: $(worker_log | tail -n 5)"
    fi
    if [ "$SECONDS" -gt "$deadline" ]; then
        fail "Timed out warming up GPUs ($warmed/$GPU_COUNT ready)"
    fi
    sleep 3
done

until curl -fsS -m 5 http://localhost:8443/health >/dev/null 2>&1; do
    if [ "$SECONDS" -gt "$deadline" ]; then
        fail "Server did not become healthy"
    fi
    sleep 2
done

report ready "$GPU_COUNT x $GPU_NAME warmed up"
