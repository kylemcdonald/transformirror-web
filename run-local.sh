#!/bin/bash

# Detect current display
CURRENT_DISPLAY=$(who | cut -d' ' -f2 | grep -E "^:[0-9]+$" | head -1)
if [ -z "$CURRENT_DISPLAY" ]; then
    CURRENT_DISPLAY=":0"
fi

echo "Using display: $CURRENT_DISPLAY"

# source venv/bin/activate
DISPLAY=$CURRENT_DISPLAY \
    __GL_SHOW_OVERLAY=0 \
    __GL_SHOW_FPS=0 \
    __GL_SHOW_GRAPHICS_OSD=0 \
    __GL_SHOW_DEBUG=0 \
    __GL_DEBUG=0 \
    $HOME/anaconda3/envs/transformirror/bin/python local_direct.py
