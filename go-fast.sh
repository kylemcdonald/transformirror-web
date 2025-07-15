#!/bin/bash

systemctl start worker
systemctl start transformirror-fast
journalctl -feu transformirror-fast