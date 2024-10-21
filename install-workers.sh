SERVICE_ID=transformirror-workers
SERVICE_NAME="transformirror workers"

USER=$(whoami)
SERVICES_DIR=/etc/systemd/system/

cat >$SERVICES_DIR/$SERVICE_ID.service <<EOL
[Unit]
Description=$SERVICE_NAME
Wants=network-online.target
After=network-online.target
[Service]
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/run-workers.sh
User=root
Restart=always
[Install]
WantedBy=multi-user.target
EOL

systemctl daemon-reload

systemctl enable $SERVICE_ID
systemctl start $SERVICE_ID