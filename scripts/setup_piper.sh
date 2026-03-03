#!/bin/bash
# 🚀 One-click setup for Piper (High performance ONNX TTS)

set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DATA_DIR="$BASE_DIR/models_data/piper-onnx"
VENV_PYTHON="$BASE_DIR/venv/bin/python3"

echo "📦 Installing Piper dependencies..."
$VENV_PYTHON -m pip install piper-tts scipy numpy

mkdir -p "$MODEL_DATA_DIR/en_US-amy-medium"
cd "$MODEL_DATA_DIR/en_US-amy-medium"

echo "📥 Downloading Amy High-Quality voice weights..."
if [ ! -f "en_US-amy-medium.onnx" ]; then
    wget -q --show-progress https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx
fi

if [ ! -f "en_US-amy-medium.onnx.json" ]; then
    wget -q --show-progress https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json
fi

echo "✅ Piper Setup Complete!"
