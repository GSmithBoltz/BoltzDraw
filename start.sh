#!/bin/sh
# Start the BoltzDraw server detached, so it survives the terminal/session
# that launched it. Logs go to server.log. Re-running is safe: if the port
# is already served, this exits without starting a second copy.
set -e
cd "$(dirname "$0")"

PORT="${PORT:-4400}"
PYTHON="${PYTHON:-/Users/geoffrey/miniconda3/envs/boltzstar/bin/python}"

if curl -s -m 2 "http://localhost:$PORT/ping" >/dev/null 2>&1; then
    echo "BoltzDraw already running on http://localhost:$PORT"
    exit 0
fi

PORT="$PORT" nohup "$PYTHON" -u server.py >> server.log 2>&1 &
echo "BoltzDraw starting on http://localhost:$PORT (pid $!, logging to server.log)"
