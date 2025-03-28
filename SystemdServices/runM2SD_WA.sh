#! /bin/bash

CLAH_PATH="/mnt/EnvsDrive/scripts_dev/_HL_Scripts/CLAH_IA"

TMUX=/usr/bin/tmux
TMUX_SOCKET_DIR="/tmp/tmux-$(id -u)"
mkdir -p "$TMUX_SOCKET_DIR"
chmod 700 "$TMUX_SOCKET_DIR"

APP_NAME="M2SD_WA"
TMUX_SERVER_FILE="$TMUX_SOCKET_DIR/$APP_NAME"

$TMUX -S "$TMUX_SERVER_FILE" start-server

# Create a new session
$TMUX -S "$TMUX_SERVER_FILE" new-session -d -s "$APP_NAME"

$TMUX -S "$TMUX_SERVER_FILE" send-keys -t "$APP_NAME" "conda activate caiman && export MKL_NUM_THREADS=1 && export OPENBLAS_NUM_THREADS=1 && export VECLIB_MAXIMUM_THREADS=1" Enter
sleep 0.5

$TMUX -S "$TMUX_SERVER_FILE" send-keys -t "$APP_NAME" "streamlit run $CLAH_PATH/CLAH_ImageAnalysis/GUI/M2SD_WA.py --server.baseUrlPath m2sd --server.port 8503" Enter

tail -f /dev/null
