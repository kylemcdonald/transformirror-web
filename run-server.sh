#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

export HF_HOME="${HF_HOME:-/workspace/.cache}"
export PYTHONUNBUFFERED=1
source .venv/bin/activate
exec python3 server.py
