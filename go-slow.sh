#!/bin/bash

echo "Starting slow mode..."
systemctl stop transformirror-fast
systemctl stop worker
sleep 2
cd /home/transformirror1/Documents/transformirror-web
bash run-local.sh
wait