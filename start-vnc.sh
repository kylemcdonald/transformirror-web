#!/bin/bash

# Detect current display
CURRENT_DISPLAY=$(who | cut -d' ' -f2 | grep -E "^:[0-9]+$" | head -1)
if [ -z "$CURRENT_DISPLAY" ]; then
    CURRENT_DISPLAY=":0"
fi

echo "Using display: $CURRENT_DISPLAY"
x11vnc -display $CURRENT_DISPLAY -forever -nopw -shared & 