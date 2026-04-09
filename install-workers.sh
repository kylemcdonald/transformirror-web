SERVICE_ID=transformirror-workers
SERVICE_NAME="transformirror workers"

USER=$(whoami)
SERVICES_DIR=/etc/systemd/system/
WORKDIR=$(pwd)
LOGDIR=$WORKDIR/logs
PIDFILE=$WORKDIR/.transformirror-workers.pid

mkdir -p "$LOGDIR"

start_without_systemd() {
    if command -v tmux >/dev/null 2>&1; then
        if tmux has-session -t "$SERVICE_ID" 2>/dev/null; then
            echo "$SERVICE_NAME already running in tmux session $SERVICE_ID"
            exit 0
        fi

        tmux new-session -d -s "$SERVICE_ID" "cd '$WORKDIR' && env HF_HOME=/workspace/.cache PYTHONUNBUFFERED=1 TRANSFORMIRROR_USE_STABLE_FAST=0 '$WORKDIR/run-workers.sh' >>'$LOGDIR/workers.log' 2>&1"
        echo "$SERVICE_ID" >"$PIDFILE"
        echo "Started $SERVICE_NAME in tmux session $SERVICE_ID"
        exit 0
    fi

    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "$SERVICE_NAME already running with PID $(cat "$PIDFILE")"
        exit 0
    fi

    nohup env \
        HF_HOME=/workspace/.cache \
        PYTHONUNBUFFERED=1 \
        TRANSFORMIRROR_USE_STABLE_FAST=0 \
        "$WORKDIR/run-workers.sh" >>"$LOGDIR/workers.log" 2>&1 &
    echo $! >"$PIDFILE"
    echo "Started $SERVICE_NAME without systemd (PID $!)"
}

cat >$SERVICES_DIR/$SERVICE_ID.service <<EOL
[Unit]
Description=$SERVICE_NAME
Wants=network-online.target
After=network-online.target
[Service]
Type=simple
WorkingDirectory=$WORKDIR
ExecStart=$WORKDIR/run-workers.sh
User=root
Restart=always
RestartSec=5
Environment=HF_HOME=/workspace/.cache
Environment=PYTHONUNBUFFERED=1
Environment=TRANSFORMIRROR_USE_STABLE_FAST=0
[Install]
WantedBy=multi-user.target
EOL

if ! command -v systemctl >/dev/null 2>&1 || [ ! -d /run/systemd/system ]; then
    start_without_systemd
    exit 0
fi

systemctl daemon-reload
systemctl enable $SERVICE_ID
systemctl start $SERVICE_ID
