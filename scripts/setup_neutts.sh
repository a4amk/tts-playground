#!/bin/bash
# 🚀 Placeholder for NeuTTS setup
echo "📦 Setting up NeuTTS dependencies..."
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PYTHON="$BASE_DIR/venv/bin/python3"

# Install system dependencies
if command -v apt-get &> /dev/null; then
    sudo apt-get update && sudo apt-get install -y espeak-ng
fi

$VENV_PYTHON -m pip install "neutts[all]" soundfile
mkdir -p "$BASE_DIR/models_data/neutts/clones"
echo "✅ NeuTTS Setup Complete!"
