#!/bin/bash
set -euo pipefail

APP_DIR="/home/oksir/Desktop/Exuvia/Exuvia-Counting-main (1)/Exuvia-Counting-main/exuvia_app"
LOG_FILE="/home/oksir/Desktop/exuvia_remote_launcher.log"

cd "$APP_DIR"

if ss -ltn 2>/dev/null | grep -q ':8501'; then
    pkill -f "streamlit run app.py" || true
    sleep 2
fi

nohup ./run.sh --headless > "$LOG_FILE" 2>&1 &