#! /bin/bash

CLAH_PATH="/mnt/EnvsDrive/scripts_dev/_HL_Scripts/CLAH_IA"

TMUX=/usr/bin/tmux
TMUX_SOCKET_DIR="/tmp/tmux-$(id -u)"
mkdir -p "$TMUX_SOCKET_DIR"
chmod 700 "$TMUX_SOCKET_DIR"

APP_NAME="SD_WA"
TMUX_SERVER_FILE="$TMUX_SOCKET_DIR/$APP_NAME"

$TMUX -S "$TMUX_SERVER_FILE" start-server

# Create a new session
$TMUX -S "$TMUX_SERVER_FILE" new-session -d -s "$APP_NAME"

$TMUX -S "$TMUX_SERVER_FILE" send-keys -t "$APP_NAME" "conda activate caiman && export MKL_NUM_THREADS=1 && export OPENBLAS_NUM_THREADS=1 && export VECLIB_MAXIMUM_THREADS=1" Enter
sleep 0.5

$TMUX -S "$TMUX_SERVER_FILE" send-keys -t "$APP_NAME" "streamlit run $CLAH_PATH/CLAH_ImageAnalysis/GUI/segDictWA.py --server.baseUrlPath segdict --server.port 8504" Enter

tail -f /dev/null
