#!/usr/bin/env bash
set -euo pipefail

SERVICE_DIR="$HOME/.config/systemd/user"
SERVICE_FILE="$SERVICE_DIR/transformirror.service"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_ID="$(id -u)"

mkdir -p "$SERVICE_DIR"

cat > "$SERVICE_FILE" <<SERVICE
[Unit]
Description=Transformirror live SDXL Turbo filter
After=graphical-session.target

[Service]
Type=simple
WorkingDirectory=$ROOT_DIR
Environment=DISPLAY=:0
Environment=XDG_RUNTIME_DIR=/run/user/$USER_ID
Environment=DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$USER_ID/bus
Environment=PYTHONNOUSERSITE=1
Environment=HF_HUB_ENABLE_HF_TRANSFER=1
ExecStart=$ROOT_DIR/run-transformirror.sh
Restart=always
RestartSec=2

[Install]
WantedBy=default.target
SERVICE

systemctl --user daemon-reload
systemctl --user enable transformirror.service
systemctl --user restart transformirror.service
systemctl --user --no-pager status transformirror.service
