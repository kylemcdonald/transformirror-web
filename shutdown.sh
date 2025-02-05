#!/bin/bash
cd "$(dirname "$0")"
systemctl stop transformirror-local
systemctl stop transformirror-workers
shutdown now