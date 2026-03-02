#!/bin/bash
set -e

echo "==============================================="
echo " Installing Chatterbox TTS "
echo "==============================================="

# 1. Activate venv
source venv/bin/activate

# 2. Install chatterbox-tts
echo "Installing pip dependencies..."
./venv/bin/python3 -m pip install chatterbox-tts onnxruntime huggingface_hub

# 3. Create models directory
mkdir -p models_data/chatterbox

echo "==============================================="
echo " Chatterbox TTS Setup Complete!"
echo " Note: Weights will be downloaded to cache automatically upon first run."
echo "==============================================="
