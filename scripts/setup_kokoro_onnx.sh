#!/bin/bash
# 🚀 One-click setup for Kokoro ONNX (82M)

set -e

BASE_DIR="/home/ubuntu/my-apps/tts-playground"
MODEL_DATA_DIR="$BASE_DIR/models_data/kokoro-onnx"
VENV_PYTHON="$BASE_DIR/venv/bin/python3"

echo "📦 Installing Kokoro ONNX dependencies..."
$VENV_PYTHON -m pip install kokoro-onnx onnxruntime numpy nest-asyncio

mkdir -p "$MODEL_DATA_DIR"
cd "$MODEL_DATA_DIR"

echo "📥 Verifying ONNX weights..."
if [ ! -f "kokoro-v0_19.onnx" ]; then
    echo "Downloading kokoro-v0_19.onnx (300MB)..."
    wget -q --show-progress https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx
fi

echo "📥 Verifying Voice Pack Metadata (bin)..."
if [ ! -f "voices.bin" ]; then
    echo "Downloading voices.bin..."
    wget -q --show-progress https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin
fi

echo "📥 Verifying Voice Pack Names (json)..."
if [ ! -f "voices.json" ]; then
    echo "Downloading voices.json..."
    wget -q --show-progress https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json
fi

echo "✅ Kokoro ONNX Setup Complete!"
