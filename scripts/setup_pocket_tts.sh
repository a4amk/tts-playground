#!/bin/bash
# 🚀 One-click setup for Pocket-TTS (Kyutai's lightweight model)

set -e

BASE_DIR="/home/ubuntu/my-apps/tts-playground"
VENV_PYTHON="$BASE_DIR/venv/bin/python3"

echo "📦 Installing Pocket-TTS dependencies..."
$VENV_PYTHON -m pip install pocket-tts beartype

echo "📥 Pre-warming Pocket-TTS weights (this will download model weights from HuggingFace)..."
# Just run a tiny generation to trigger the download
$VENV_PYTHON -c "from pocket_tts.models.tts_model import TTSModel; model = TTSModel.load_model('b6369a24'); print('Weights loaded successfully!')"

echo "✅ Pocket-TTS Setup Complete!"
