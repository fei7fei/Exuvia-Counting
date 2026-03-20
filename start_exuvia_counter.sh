#!/bin/bash
set -euo pipefail

APP_DIR="/home/oksir/Desktop/Exuvia/Exuvia-Counting-main (1)/Exuvia-Counting-main/exuvia_app"

echo "Exuvia Counter launcher"
echo "Status: STARTING"
echo "App directory: $APP_DIR"

cd "$APP_DIR"

if ss -ltn 2>/dev/null | grep -q ':8501'; then
	echo "Existing app detected on port 8501"
	echo "Stopping old app instance"
	pkill -f "streamlit run app.py" || true
	sleep 2
fi

echo "Starting fresh app instance"
./run.sh
