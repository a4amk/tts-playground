#!/bin/bash
set -e

echo "==============================================="
echo " Installing Chatterbox Turbo ONNX (CPU)        "
echo "==============================================="

# 1. Activate venv
source venv/bin/activate

# 2. Install dependencies
echo "Installing pip dependencies..."
./venv/bin/python3 -m pip install onnxruntime transformers huggingface_hub librosa soundfile resemble-perth

# 3. Create models directory
mkdir -p models_data/chatterbox_onnx

echo "==============================================="
echo " Chatterbox Turbo ONNX Setup Complete!"
echo " Note: Weights will be downloaded to cache automatically upon first run."
echo "==============================================="
