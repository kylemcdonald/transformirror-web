#!/bin/bash
# Container start hook for pods launched by transformirror-control.
# The pod's start command downloads this script, runs it in the background,
# and then hands off to the image's normal /start.sh (SSH, etc).
set -uo pipefail

REPO=${TRANSFORMIRROR_REPO:-https://github.com/kylemcdonald/transformirror-web.git}
REF=${TRANSFORMIRROR_REF:-main}
DIR=/workspace/transformirror-web

report() {
    echo "[bootstrap] $(date -u +%H:%M:%S) $1${2:+: $2}"
    if [ -n "${CONTROL_HOOK_URL:-}" ] && [ -n "${CONTROL_HOOK_TOKEN:-}" ]; then
        body=$(python3 -c 'import json, sys; print(json.dumps({"stage": sys.argv[1], "detail": sys.argv[2]}))' "$1" "${2:-}")
        curl -fsS -m 10 -X POST "$CONTROL_HOOK_URL" \
            -H "Authorization: Bearer $CONTROL_HOOK_TOKEN" \
            -H "Content-Type: application/json" \
            --data "$body" >/dev/null 2>&1 || true
    fi
}

report container "$(nvidia-smi -L | wc -l) GPU(s): $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"

mkdir -p /workspace
if [ -d "$DIR/.git" ]; then
    git -C "$DIR" fetch --depth 1 origin "$REF" && git -C "$DIR" reset --hard FETCH_HEAD
else
    git clone --depth 1 --branch "$REF" "$REPO" "$DIR"
fi || { report failed "Could not clone $REPO@$REF"; exit 1; }

report cloned "$(git -C "$DIR" log -1 --format='%h %s')"

mkdir -p "$DIR/logs"
exec "$DIR/setup-runpod.sh" >> "$DIR/logs/setup.log" 2>&1
