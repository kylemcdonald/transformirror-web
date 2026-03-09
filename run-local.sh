#!/bin/bash

PYTHON_BIN=/home/bizon/anaconda3/envs/transformirror-py31020-clean/bin/python
USER_ID=$(id -u)
export DISPLAY="${DISPLAY:-:0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$USER_ID}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}"
export PULSE_SERVER="${PULSE_SERVER:-unix:$XDG_RUNTIME_DIR/pulse/native}"
export SDL_AUDIODRIVER="${SDL_AUDIODRIVER:-pulse}"
export PYTHONNOUSERSITE=1

for _ in $(seq 1 15); do
    [ -S "$XDG_RUNTIME_DIR/pulse/native" ] && break
    sleep 1
done

# source venv/bin/activate
$PYTHON_BIN local.py
