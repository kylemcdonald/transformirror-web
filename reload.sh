#!/bin/bash
cd "$(dirname "$0")"
systemctl stop transformirror
sleep 5
systemctl start transformirror