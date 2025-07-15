#!/bin/bash

echo "Starting fast mode..."
systemctl start worker
sleep 10
systemctl start transformirror-fast
journalctl -feu transformirror-fast