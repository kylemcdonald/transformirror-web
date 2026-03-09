SERVICE_ID=transformirror-local
SERVICE_NAME="transformirror local"

USER=bizon
SERVICES_DIR=/etc/systemd/system/

sudo tee $SERVICES_DIR/$SERVICE_ID.service >/dev/null <<EOL
[Unit]
Description=$SERVICE_NAME
Wants=network-online.target
After=network-online.target graphical.target
[Service]
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/run-local.sh
User=$USER
Environment=DISPLAY=:0
Environment=XDG_RUNTIME_DIR=/run/user/1000
Environment=DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
Environment=PULSE_SERVER=unix:/run/user/1000/pulse/native
Restart=always
[Install]
WantedBy=multi-user.target
EOL

sudo systemctl daemon-reload

sudo systemctl enable $SERVICE_ID
sudo systemctl start $SERVICE_ID
