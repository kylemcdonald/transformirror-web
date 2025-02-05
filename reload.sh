#!/bin/bash
cd "$(dirname "$0")"
systemctl stop transformirror-local
systemctl restart transformirror-workers
systemctl start transformirror-local