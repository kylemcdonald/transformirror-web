#!/usr/bin/env bash
# Installs odaiba.service as a systemd user service. The machine auto-logs in
# to its X11 desktop on boot, which starts graphical-session.target and with it
# this service. It restarts automatically if it exits.
set -euo pipefail

SERVICE_DIR="$HOME/.config/systemd/user"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SERVICE_DIR"

cat > "$SERVICE_DIR/odaiba.service" <<SERVICE
[Unit]
Description=Odaiba live img2img LED wall
PartOf=graphical-session.target
After=graphical-session.target

[Service]
Type=simple
WorkingDirectory=$ROOT_DIR
ExecStart=$ROOT_DIR/run-odaiba.sh
Restart=always
RestartSec=3

[Install]
WantedBy=graphical-session.target
SERVICE

systemctl --user daemon-reload
systemctl --user enable odaiba.service
systemctl --user restart odaiba.service
systemctl --user --no-pager status odaiba.service | head -5
