# 🚂 Engines Overview

The TTS Playground supports a variety of engines, each with unique strengths. This guide provides a technical overview of each plugin.

| Engine ID | Display Name | Language Support | Zero-Shot Cloning | Latency |
|-----------|--------------|------------------|-------------------|---------|
| `kokoro` | Kokoro v1.0 (PyTorch) | English | ❌ No | ⚡ Ultra Low |
| `kokoro-onnx` | Kokoro v0.19 (ONNX) | English, Japanese | ❌ No | 🚀 Very Low |
| `chatterbox_onnx`| Chatterbox Turbo | English | ✅ Yes | ⚡ Low |
| `genie` | Genie (GPT-SoVITS) | EN, JA, ZH | ✅ Yes | 🐢 Medium |
| `zipvoice` | ZipVoice | English | ✅ Yes | 🚀 Low |
| `piper` | Piper (VITS) | English (Amy) | ❌ No | ⚡ Ultra Low |
| `pocket_tts` | Pocket-TTS (Mimi) | English | ❌ No | ⚡ Low |
| `neutts` | NeuTTS | English | ✅ Yes | ⚡ Low |

---

## 🎭 Engine Deep-Dives

### 1. **Kokoro (PyTorch & ONNX)**
- **Best For**: High-quality natural narration with minimal CPU/GPU overhead.
- **Languages**: Primarily English. ONNX version supports Japanese.
- **Assets**: `models_data/kokoro-82m/` or `models_data/kokoro-onnx/`.

### 2. **Chatterbox Turbo (ONNX)**
- **Best For**: Real-time conversational AI and zero-shot voice cloning.
- **Cloning**: Supports cloning from a voice reference (< 15s recommended).
- **Controls**: Includes **Exaggeration** and **Repetition Penalty**.

### 3. **Genie (GPT-SoVITS)**
- **Best For**: High-fidelity cloning across English, Japanese, and Chinese.
- **Cloning**: Requires a voice reference and a corresponding transcript.
- **Note**: Language selection is critical for proper prosody.

### 4. **ZipVoice**
- **Best For**: Low-latency distilled flow-matching synthesis.
- **Cloning**: Supports references. Optimized for quantized ONNX runtimes.

### 5. **NeuTTS**
- **Best For**: High-performance synthesis with easy cloning.
- **Storage**: Clones are saved as JSON metadata (codes + text) in `custom_voices/neutts/`.

### 6. **Piper (VITS)**
- **Best For**: Extremely fast edge synthesis. Uses high-quality ONNX exported VITS models.

---

## 📂 Standardized Directories

- **Weights**: All static weights must reside in `models_data/`.
- **Custom Voices**: All dynamically saved clones must reside in `custom_voices/<engine_id>/`.
- **Logs**: Generation logs are piped to the terminal and `server.log`.
