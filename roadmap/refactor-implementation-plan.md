# Implementation Plan: Plug-and-Play TTS Architecture

This document breaks down the refactoring effort into systemic phases.

## Phase 1: The Foundation (Interfaces & Managers)
Before touching any existing models or UI files, we establish the strict contract that all future models must obey.
1. **Create `app/engines/interface.py`**
   - Define the `TTSPlugin` abstract base class.
   - Outline strict abstract methods: `generate_stream`, `generate_batch`, `load`, `get_ui_config`, `get_cloning_config`, `save_clone`, and `list_clones`.
   - Add dependency management properties: `is_installed()` and `install_dependencies()` to expose the plugin's required python packages.
2. **Create `app/engines/manager.py`**
   - Write a `PluginManager` class that uses `importlib` and `os.walk` or explicit registration paths to load engine dictionaries safely.
   - Implement caching so the manager doesn't rescan the disk on every UI refresh.
3. **Establish Global Configuration**
   - Create a `config.yaml` or `.env` system to allow users to toggle whether models use global `huggingface_hub` caching or download directly into their local `app/engines/<plugin_name>/weights/` directory.
   - Enforce a strict `.gitignore` policy that natively ignores the `custom_voices/` directory and any localized `weights/` subdirectories to keep the repository clean.

## Phase 2: Refactoring Existing Engines
Convert every existing engine one by one to use the `TTSPlugin` standard.
1. **Kokoro & Kokoro ONNX:** Standardize the fastest local engines.
2. **Piper:** Port the native VITS implementation and its 22kHz resampling logic.
3. **Chatterbox Turbo ONNX:** Localize the zero-shot `.wav` saving functionality.
4. **Genie:** Localize the multi-file cloning logic (`.wav`, `.txt`, `.base`).
5. **ZipVoice:** Localize the exact-transcript cloning constraint.
6. **Pocket-TTS:** Localize the `.pt` and `.safetensors` state export logic.

**Requirement During conversion:** Strip all engine-specific download scripts (`scripts/setup_...sh`) and bake the weight fetching directly into the new `load()` methods using safe, asynchronous `huggingface_hub` pulls.

## Phase 3: UI & WebSocket Decoupling
Wire the generalized backend systems into the frontend cleanly.
1. **WebSocket Handler (`app/api/ws.py`)**
   - Replace the static `models.get()` registry lookups with `PluginManager.get_plugin(model_id)`.
   - Ensure the engine's `load()` method is triggered cleanly when the first text chunk hits the socket buffer.
2. **Gradio UI Rebuild (`app/ui/gradio_app.py`)**
   - Populate the Model Selection dropdown dynamically using `PluginManager.get_all_engine_ids()`.
   - Incorporate a 1-Click Installation UI to check `plugin.is_installed()` when selected. If false, display an "Install Dependencies" button that triggers `plugin.install_dependencies()`.
   - Replace the massive `if/else` block in `update_cloning_ui()` by querying the selected plugin's `get_cloning_config()`.
   - Replace the dictionary in `update_sliders_for_model()` with `plugin.get_ui_config()`.
   - Update the submit clone handler to dynamically route the uploaded audio and text to the selected plugin's `save_clone()` method.

## Phase 4: Teardown & Validation
1. Safely delete `app/engines/registry.py`.
2. Safely delete `app/engines/clones.py`.
3. Safely delete the global `models_data/` directory. All weights should now be natively managed by their respective plugins (e.g., `app/engines/<plugin>/weights/` or via huggingface cache).
4. Conduct end-to-end testing of every supported model. Validate Streaming latency (TTFB/RTF metrics) and Batch Synthesis generation.
5. Update developer documentation on how to build a custom `TTSPlugin`.
