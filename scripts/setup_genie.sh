#!/bin/bash
# 🚀 One-click setup for Genie (GPT-SoVITS)
set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DATA_DIR="$BASE_DIR/models_data/genie"
VENV_PYTHON="$BASE_DIR/venv/bin/python3"

echo "==============================================="
echo " Installing Genie (GPT-SoVITS) dependencies    "
echo "==============================================="

$VENV_PYTHON -m pip install torch numpy scipy librosa transformers huggingface_hub genie-tts

mkdir -p "$MODEL_DATA_DIR"

echo "📥 Verifying GenieData..."
# Use the internal downloader if missing
$VENV_PYTHON -c "from genie_tts import download_genie_data; import os; os.environ['GENIE_DATA_DIR'] = '$MODEL_DATA_DIR/GenieData'; download_genie_data('$MODEL_DATA_DIR/GenieData')"

echo "==============================================="
echo " Genie Setup Complete!                         "
echo " Weights are located in: $MODEL_DATA_DIR/GenieData"
echo "==============================================="
