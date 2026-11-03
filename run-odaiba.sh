#!/usr/bin/env bash
# Launches the Odaiba installation (camera -> img2img -> LED wall). No audio.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_ID="$(id -u)"

export DISPLAY="${DISPLAY:-:0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/${USER_ID}}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=${XDG_RUNTIME_DIR}/bus}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

# Keep the displays awake.
gsettings set org.gnome.desktop.session idle-delay 0 >/dev/null 2>&1 || true
gsettings set org.gnome.desktop.screensaver lock-enabled false >/dev/null 2>&1 || true
gsettings set org.gnome.desktop.screensaver idle-activation-enabled false >/dev/null 2>&1 || true
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing' >/dev/null 2>&1 || true
xset s off s noblank -dpms >/dev/null 2>&1 || true

cd "$ROOT_DIR"
exec "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/odaiba_live.py" "$@"
