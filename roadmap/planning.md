# 🗺️ Universal TTS Playground Roadmap

## ✅ Phase 1: Core Foundation & Kokoro 
- [x] Initial FastAPI + Gradio Architecture
- [x] Low-Latency WebSocket Streaming
- [x] Native JS Audio Engine integration
- [x] Kokoro-82M (PyTorch) Engine
- [x] Kokoro-82M (ONNX) Engine (Optimized for CPU)
- [x] Central Engine Registry system
- [x] Real-time metrics (TTFB, RTF) UI

## ✅ Phase 2: Performance & Variety
- [x] **Piper (ONNX)**: Native VITS-based engine with high performance and 22kHz resampling logic.
- [x] **Roadmap & Planning Documentation**: Track project progress.
- [x] **Advanced Controls**: Integrated Temp, Top-K, Top-P, Repetition Penalty, Seed.
- [x] Smart Model Presets: UI automatically updates sliders with model-optimized defaults.
- [x] **Repository Cleanup**: Organized test scripts and removed obsolete files.
- [x] **Git Initialization**: Configured user info, license, and .gitignore.

## ✅ Phase 3: Advanced Cloning & Local Optimization
- [x] **Pocket TTS**: Ultra-lightweight emotional TTS.
- [x] **Zip Voice**: High-quality local voice models (Distill version).
- [x] **Voice Upload Interface**: UI for managing custom reference samples for cloning (Chatterbox, Genie, ZipVoice, Pocket-TTS).
- [x] **One-Click Setup Scripts**: Standardized for every newly added engine.

## 🚀 Phase 4: High Fidelity Zero-Shot CPU ONNX Variants
- [ ] **Pocket-TTS Variants**: Add INT8 and FP32 ONNX variants of Kyutai's Pocket-TTS. *(Note: Exhaustive search confirms Kyutai Pocket-TTS is strictly trained on English data natively; no Hindi or Urdu weights currently exist on HuggingFace out of the box).*
- [x] **Chatterbox Turbo English**: Integrate the Chatterbox Turbo ONNX framework.
- [x] **NeuTTS**: High-fidelity neural voice engine for ultra-clean speech synthesis (Initial Setup Complete).

- [ ] **F5-TTS CPU**: Adapt F5-TTS to run in real-time inference on the CPU pipeline.

## 📱 Phase 5: Cross-Platform Streaming
- [ ] **Standardized Test Suite**: Implement a robust unit and integration testing framework using `pytest`, ensuring all engines and API endpoints are covered.
- [ ] **Safari / iOS Streaming Compatibility**: Make real-time WebSocket audio streaming work natively on Safari and iOS devices (handling strict AudioContext constraints and buffer formats).

## 🏔️ Long-Term Goal
Build the most accessible, developer-friendly local TTS laboratory for rapid model benchmarking and deployment.
