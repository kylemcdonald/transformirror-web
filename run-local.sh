#!/bin/bash

# source venv/bin/activate
DISPLAY=:0 \
    __GL_SHOW_OVERLAY=0 \
    __GL_SHOW_FPS=0 \
    __GL_SHOW_GRAPHICS_OSD=0 \
    __GL_SHOW_DEBUG=0 \
    __GL_DEBUG=0 \
    $HOME/anaconda3/envs/transformirror/bin/python local.py
