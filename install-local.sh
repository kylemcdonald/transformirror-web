SERVICE_ID=transformirror-local
SERVICE_NAME="transformirror local"

USER=bizon
SERVICES_DIR=/etc/systemd/system/

sudo cat >$SERVICES_DIR/$SERVICE_ID.service <<EOL
[Unit]
Description=$SERVICE_NAME
Wants=network-online.target
After=network-online.target
[Service]
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/run-local.sh
User=$USER
Restart=always
[Install]
WantedBy=multi-user.target
EOL

sudo systemctl daemon-reload

sudo systemctl enable $SERVICE_ID
sudo systemctl start $SERVICE_ID