#!/bin/bash

SERVICE_ID=transformirror-local
SERVICE_NAME="transformirror local"

USER=transformirror1
SERVICES_DIR=/etc/systemd/system/

# Use sudo tee to write directly to the protected directory
sudo tee "$SERVICES_DIR/$SERVICE_ID.service" > /dev/null <<EOL
[Unit]
Description=$SERVICE_NAME
Wants=network-online.target
After=network-online.target
Wants=graphical-session.target
After=graphical-session.target
[Service]
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/run-local.sh
User=$USER
Group=$USER
Restart=always
Environment=DISPLAY=:0
Environment=XAUTHORITY=/home/$USER/.Xauthority
Environment=PULSE_RUNTIME_PATH=/run/user/$(id -u $USER)/pulse
Environment=XDG_RUNTIME_DIR=/run/user/$(id -u $USER)
[Install]
WantedBy=multi-user.target
EOL

sudo systemctl daemon-reload

sudo systemctl enable $SERVICE_ID
# sudo systemctl start $SERVICE_ID