#!/bin/bash
# 🚀 One-click setup for Genie (GPT-SoVITS)
set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DATA_DIR="$BASE_DIR/models_data/genie"
VENV_PYTHON="$BASE_DIR/venv/bin/python3"

echo "==============================================="
echo " Installing Genie (GPT-SoVITS) dependencies    "
echo "==============================================="

$VENV_PYTHON -m pip install torch numpy scipy librosa transformers huggingface_hub

mkdir -p "$MODEL_DATA_DIR/GenieData"
mkdir -p "$MODEL_DATA_DIR/characters"

echo "==============================================="
echo " Genie Setup Complete!                         "
echo " Note: Weights are expected in models_data/genie"
echo "==============================================="
