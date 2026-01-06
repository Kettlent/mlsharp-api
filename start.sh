#!/bin/bash

### -------------------------
### ML-SHARP Auto-Start Script
### -------------------------

LOG_FILE="/workspace/sharp.log"
ENV_NAME="sharp"
SERVER_DIR="/workspace/mlsharp-api"   # 🔁 adjust only if repo path differs
PORT=8000

echo "" >> $LOG_FILE
echo "-----------------------------------" >> $LOG_FILE
echo "Starting ML-SHARP Server at $(date)" >> $LOG_FILE
echo "-----------------------------------" >> $LOG_FILE

# Load conda
if [ -f /workspace/miniconda/etc/profile.d/conda.sh ]; then
    echo "[OK] Loading conda..." >> $LOG_FILE
    source /workspace/miniconda/etc/profile.d/conda.sh
else
    echo "[ERROR] conda.sh not found!" >> $LOG_FILE
    exit 1
fi

# Activate environment
echo "[OK] Activating conda env: $ENV_NAME" >> $LOG_FILE
conda activate $ENV_NAME

# Kill old server if exists
OLD_PIDS=$(pgrep -f "uvicorn sharp.api.main:app")
if [ ! -z "$OLD_PIDS" ]; then
    echo "[INFO] Killing old SHARP server processes: $OLD_PIDS" >> $LOG_FILE
    pkill -f "uvicorn sharp.api.main:app"
fi

# Go to server directory
cd $SERVER_DIR || {
    echo "[ERROR] Could not cd to $SERVER_DIR" >> $LOG_FILE
    exit 1
}

echo "[OK] Starting ML-SHARP server on port $PORT..." >> $LOG_FILE

# Start server in background
nohup uvicorn sharp.api.main:app \
    --host 0.0.0.0 \
    --port $PORT \
    >> $LOG_FILE 2>&1 &

sleep 3

# Verify server started
NEW_PID=$(pgrep -f "uvicorn sharp.api.main:app")
if [ -z "$NEW_PID" ]; then
    echo "[ERROR] ML-SHARP server failed to start!" >> $LOG_FILE
else
    echo "[OK] ML-SHARP server running with PID: $NEW_PID" >> $LOG_FILE
fi

echo "-----------------------------------" >> $LOG_FILE
echo "ML-SHARP Startup Complete" >> $LOG_FILE
echo "-----------------------------------" >> $LOG_FILE
