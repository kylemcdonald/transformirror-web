#!/bin/bash

SERVICE_ID=daily-shutdown
SERVICE_NAME="Daily Scheduled Shutdown"
SERVICES_DIR=/etc/systemd/system/
SHUTDOWN_TIME="22:05"

# --- Create the service ---
sudo tee "${SERVICES_DIR}/${SERVICE_ID}.service" > /dev/null <<EOL
[Unit]
Description=$SERVICE_NAME

[Service]
Type=oneshot
ExecStart=/usr/bin/systemctl poweroff
EOL

# --- Create the timer ---
sudo tee "${SERVICES_DIR}/${SERVICE_ID}.timer" > /dev/null <<EOL
[Unit]
Description=Trigger daily shutdown at $SHUTDOWN_TIME

[Timer]
OnCalendar=*-*-* $SHUTDOWN_TIME
Persistent=true

[Install]
WantedBy=timers.target
EOL

# --- Enable & reload ---
sudo systemctl daemon-reload
sudo systemctl enable --now "${SERVICE_ID}.timer"

echo "--------------------------------------------------"
echo "Installed:"
echo "  ${SERVICE_ID}.service"
echo "  ${SERVICE_ID}.timer"
echo
echo "The system will now shut down every day at $SHUTDOWN_TIME."
echo "--------------------------------------------------"