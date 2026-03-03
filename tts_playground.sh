#!/bin/bash

# tts_playground.sh
# Unified management script for the TTS Playground

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

PID_FILE="server.pid"

show_help() {
    echo "Usage: ./tts_playground.sh [OPTION]"
    echo ""
    echo "Options:"
    echo "  --install    Install prerequisites and setup virtual environment"
    echo "  --start      Start the TTS Playground server"
    echo "  --stop       Stop the running TTS Playground server"
    echo "  --purge      Delete localized model weights, logs, and venv"
    echo "  --help       Show this help message"
}

do_install() {
    echo ">>> Installing prerequisites..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update && sudo apt-get install -y espeak-ng python3-venv
    fi

    echo ">>> Setting up virtual environment..."
    python3 -m venv venv
    ./venv/bin/pip install --upgrade pip
    ./venv/bin/pip install -r requirements.txt
    
    echo ">>> Installation complete."
}

do_start() {
    if [ ! -d "venv" ]; then
        echo "Error: Virtual environment not found. Please run ./tts_playground.sh --install first."
        exit 1
    fi

    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null; then
            echo "Server is already running (PID: $PID)."
            exit 0
        else
            rm "$PID_FILE"
        fi
    fi

    echo ">>> Starting TTS Playground..."
    # We run in the foreground by default, but store PID if we ever want to background it
    ./venv/bin/python3 main.py &
    echo $! > "$PID_FILE"
    wait $!
    rm "$PID_FILE"
}

do_stop() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        echo ">>> Stopping TTS Playground (PID: $PID)..."
        kill $PID
        rm "$PID_FILE"
    else
        # Fallback: find and kill main.py
        PIDS=$(pgrep -f "python3 main.py")
        if [ -n "$PIDS" ]; then
            echo ">>> Stopping matching processes: $PIDS"
            kill $PIDS
        else
            echo "No server process found."
        fi
    fi
}

do_purge() {
    echo "WARNING: This will delete all model weights in models_data/, logs, and the venv."
    read -p "Are you sure? (y/N): " confirm
    if [[ "$confirm" == [yY] || "$confirm" == [yY][eE][sS] ]]; then
        echo ">>> Purging data..."
        rm -rf models_data/*
        rm -rf venv/
        rm -rf *.log
        rm -rf server.pid
        echo ">>> Purge complete."
    else
        echo "Purge cancelled."
    fi
}

case "$1" in
    --install)
        do_install
        ;;
    --start)
        do_start
        ;;
    --stop)
        do_stop
        ;;
    --purge)
        do_purge
        ;;
    --help|*)
        show_help
        ;;
esac
