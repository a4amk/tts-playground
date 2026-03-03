# A Pluggable TTS Playground

**A high-performance, extensible Text-to-Speech (TTS) playground and API server designed for low-latency streaming and maximum generation control.** 

This project provides a unified interface for state-of-the-art TTS models, optimized for real-time applications and developer-centric voice synthesis.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-tts--playground-blue?logo=github)](https://github.com/a4amk/tts-playground)

---

![TTS Interface](https://public.a4amk.com/projects/tts-playground/screenshot-tts-playground.png) <!-- Screenshot -->

## Key Features

*   **Low-Latency WebSocket Streaming**: Uses a native JavaScript `AudioContext` and WebSockets to stream audio bytes directly to the browser hardware buffer, bypassing Gradio's internal chunking bottlenecks.
*   **Instant Buffer Purging**: Clicking "Stop" immediately kills the WebSocket and clears the browser's audio queue—no "leftover" words playing after you stop.
*   **Advanced Synthesis Controls**: Granular control over Temperature, Top-K, Top-P, Repetition Penalty, and Seed.
*   **Dynamic Plugin Discovery**: Simply drop a folder into `app/engines/` to add new models. No hardcoded registries.
*   **High-Fidelity Zero-Shot Cloning**: Support for Pocket TTS, Chatterbox Turbo, ZipVoice, NeuTTS, and Genie with drag-and-drop reference audio uploading.
*   **Real-Time Metrics**: Live tracking of TTFB (Time To First Byte) and RTF (Real-Time Factor) directly in the UI.
*   **Maximum Chunking Control**: Select from predefined regex patterns (Sentences, Newlines, Words) or provide your own custom regex for precise partitioning.

---

## Quick Start

### 1. Prerequisites
Ensure you have Python 3.10+, `python3-venv`, and `espeak-ng` installed on your system.
```bash
# Linux (Ubuntu/Debian)
sudo apt-get update && sudo apt-get install -y espeak-ng python3-venv
```

### 2. Installation
Clone the repository and install dependencies into a virtual environment:
```bash
git clone https://github.com/a4amk/tts-playground.git
cd tts-playground
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

### 3. Run the App
Start the unified FastAPI + Gradio server:
```bash
./venv/bin/python3 main.py
```
Navigate to `http://localhost:7860` to access the UI.

---

## Project Structure

```plaintext
.
├── main.py              # Application entrypoint
├── app/
│   ├── api/             # FastAPI WebSocket logic
│   ├── engines/         # Self-contained Plugin Engines
│   └── ui/              # Gradio interface & JS snippets
├── custom_voices/       # Standardized directory for voice clones
├── models_data/         # Unified storage for weights & assets
├── scripts/             # One-click installation scripts
├── docs/                # Technical deep-dives & architecture
└── tests/               # Pytest integration suite
```

---

## One-Click Model Setup

Each engine in the playground has a dedicated setup script to handle dependency installation and weight downloads. All models are maintained in their original form and should work correctly with their natively supported languages.

*   **Kokoro PyTorch**: `bash scripts/setup_kokoro_pytorch.sh`
*   **Kokoro ONNX**: `bash scripts/setup_kokoro_onnx.sh`
*   **Piper (VITS)**: `bash scripts/setup_piper.sh`
*   **Pocket-TTS**: `bash scripts/setup_pocket_tts.sh`
*   **ZipVoice**: `bash scripts/setup_zipvoice.sh`
*   **NeuTTS**: `bash scripts/setup_neutts.sh`
*   **Chatterbox Turbo**: `bash scripts/setup_chatterbox_onnx.sh`
*   **Genie (GPT-SoVITS)**: `bash scripts/setup_genie.sh`

> [!TIP]
> All models are optimized for English by default. Use the setup scripts to ensure all required model assets are placed correctly in `models_data/`.

---

## Supported Models and Credits

This playground integrates several state-of-the-art open-source TTS engines:

*   **Kokoro-82M**: A highly efficient 82M parameter model. [Reference](https://huggingface.co/hexgrad/Kokoro-82M)
*   **Chatterbox Turbo**: High-speed conversational TTS by Resemble AI. [Reference](https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX)
*   **Piper**: A fast, local neural text to speech system. [Reference](https://github.com/rhasspy/piper)
*   **ZipVoice**: High-speed speech synthesis with distilled flow matching. [Reference](https://github.com/k2-fsa/ZipVoice)
*   **Genie (GPT-SoVITS)**: Powerful few-shot voice cloning. [Reference](https://github.com/RVC-Boss/GPT-SoVITS)
*   **Pocket-TTS**: Lightweight TTS by Kyutai Labs. [Reference](https://github.com/kyutai-labs/pocket-tts)
*   **NeuTTS**: High-performance neural TTS by Neuphonic. [Reference](https://github.com/neuphonic/neutts)

---

## Adding New Models

The architecture is built for rapid extension. To add a new model:
1. Create a folder in `app/engines/`.
2. Define an `engine.py` inheriting from `app.engines.interface.TTSPlugin`.
3. The server will automatically discover and load your new engine at startup.

See [docs/adding_models.md](./docs/adding_models.md) for a full guide.

---

## Metrics Definition

*   **TTFA (Cold Start)**: Time To First Audio. Measured from the moment the "Start" button is clicked to the absolute first audio sample hitting the JS AudioContext. Includes initial model loading and setup overhead. **Lower is better**.
*   **TTFA (Warm Start)**: Measured from the moment the "Start" button is clicked to the absolute first audio sample, excluding the model loading time. This reflects pure inference latency in a pre-warmed production environment. **Lower is better**.
*   **RTF (Real-Time Factor)**: (Generation Time / Audio Duration). A value of `0.5x` means the model generated 10 seconds of speech in 5 seconds. **Lower is better**.

---

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
