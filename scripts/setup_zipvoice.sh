#!/bin/bash
# 🚀 One-click setup for ZipVoice

set -e

BASE_DIR="/home/ubuntu/my-apps/tts-playground"
MODEL_DATA_DIR="$BASE_DIR/models_data/zipvoice"
VENV_PYTHON="$BASE_DIR/venv/bin/python3"

echo "📦 Cloning ZipVoice repository..."
mkdir -p "$MODEL_DATA_DIR"
if [ ! -d "$MODEL_DATA_DIR/ZipVoice" ]; then
    git clone https://github.com/k2-fsa/ZipVoice.git "$MODEL_DATA_DIR/ZipVoice"
fi

cd "$MODEL_DATA_DIR/ZipVoice"

echo "📦 Installing ZipVoice dependencies..."
$VENV_PYTHON -m pip install -r requirements.txt
$VENV_PYTHON -m pip install onnxruntime

echo "📥 Downloading Pre-trained weights..."
# They usually download via huggingface, let's just make sure hf_hub_download works inside their code.
# The user might need a specific model folder.
mkdir -p "$MODEL_DATA_DIR/models"

echo "✅ ZipVoice Setup Complete!"
