#!/bin/bash
# 🚀 One-click setup for Kokoro PyTorch (82M)

set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DATA_DIR="$BASE_DIR/models_data/kokoro-82m"
VENV_PYTHON="$BASE_DIR/venv/bin/python3"

echo "📦 Installing Kokoro PyTorch dependencies..."
$VENV_PYTHON -m pip install kokoro soundfile numpy

mkdir -p "$MODEL_DATA_DIR"
cd "$MODEL_DATA_DIR"

echo "📥 Verifying Model Weights..."
if [ ! -f "kokoro-v1_0.pth" ]; then
    echo "Downloading kokoro-v1_0.pth (300MB)..."
    wget -q --show-progress https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1_0.pth
fi

echo "📥 Verifying Voice Packs..."
if [ ! -d "voices" ]; then
    echo "Downloading voices.tar.gz..."
    wget -q --show-progress https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices.tar.gz
    tar -xzf voices.tar.gz
    rm voices.tar.gz
fi

echo "✅ Kokoro PyTorch Setup Complete!"
