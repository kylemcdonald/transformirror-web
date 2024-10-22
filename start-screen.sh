#!/bin/bash
apt update
apt install -y screen
screen -S workers -dm bash -c './run-workers.sh; exec bash'
screen -S server -dm bash -c './run-server.sh; exec bash'