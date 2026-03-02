# Refactor Progress Tracker

Use this document to track granular progress throughout the Plug-and-Play architecture refactor.

## Phase 1: Foundation
- [ ] Define `TTSPlugin` abstract interface (`interface.py`)
- [ ] Implement UI Defaults mapping in Interface
- [ ] Implement Cloning Specs mapping in Interface 
- [ ] Build `PluginManager` for dynamic module discovery (`manager.py`)
- [ ] Setup `config.yaml` for weight storage toggles (HF Cache vs Local).
- [ ] Construct comprehensive `.gitignore` for `custom_voices/` and `weights/`.

## Phase 2: Engine Conversion
- [ ] Refactor: **Kokoro (PyTorch)**
- [ ] Refactor: **Kokoro (ONNX)**
- [ ] Refactor: **Piper**
- [ ] Refactor: **Chatterbox Turbo ONNX** (including local voice saving)
- [ ] Refactor: **Genie** (including base-model dependency cloning)
- [ ] Refactor: **ZipVoice** (including transcript-enforced cloning)
- [ ] Refactor: **Pocket-TTS** (including safetensors cloning layer)
- [ ] **Validation:** Ensure all setup scripts are deprecated and downloading is natively handled in `load()`.

## Phase 3: UI & System Decoupling
- [ ] Hook WebSocket API (`ws.py`) strictly to `PluginManager`.
- [ ] Refactor Gradio Model Dropdown to populate dynamically.
- [ ] Implement `Install Dependencies` button UI logic (`gradio_app.py`) dynamically checking `plugin.is_installed()`.
- [ ] Refactor UI Sliders to update via plugin property callbacks.
- [ ] Refactor Voice Cloning UI to draw instructions directly from plugin definitions.
- [ ] Reroute the actual UI upload/save action to trigger `plugin.save_clone()`.

## Phase 4: Cleanup
- [ ] Delete `registry.py`
- [ ] Delete `clones.py`
- [ ] Safely delete `models_data/` after all weights are decentralized.
- [ ] **Standardize Test Suite**: Migrate legacy scripts to `pytest`.
- [ ] Perform final real-time stream stress test.
- [ ] Update `README.md` to reflect new plugin architecture.
