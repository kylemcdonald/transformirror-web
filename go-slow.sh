#!/bin/bash

systemctl stop worker
systemctl stop transformirror-fast
cd /home/transformirror1/Documents/transformirror-web
bash run-local.sh
wait