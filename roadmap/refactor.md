# Universal TTS Playground: Plugin Architecture Refactor

## Objective
Transition the current hardcoded, multi-file monolithic architecture into a **Decentralized Plug-and-Play System**. This allows new TTS engines to be added, tested, and distributed as isolated folders without altering core project files like `registry.py`, `clones.py`, or `gradio_app.py`.

## Core Requirements & Tenets

### 1. Zero-Touch Core System
The act of adding or removing a model must be entirely self-contained. Deleting a directory from `app/engines/` should safely and cleanly remove it from the UI, the WebSocket stream, and the file system without throwing unhandled exceptions in the core application.

### 2. Auto-Discovery & Lazy Loading
- **Auto-Discovery:** Upon boot, a central `PluginManager` scans the `app/engines/` directory for valid `TTSPlugin` subclasses.
- **Lazy Initialization:** To keep RAM usage minimal, models **must not** load their ML weights or initialize ONNX sessions until a user actively requests synthesis from that specific engine.
- **On-Demand Downloads:** Replace standalone `.sh` scripts. Engines should use mechanisms like `huggingface_hub` to verify and download missing weights immediately prior to their first `load()`.
- **1-Click Installation:** Plugins can define a `requirements.txt` or equivalent pip dependencies. The UI will expose a 1-click installation button to trigger `pip install` transparently if these modules are missing.
- **Localized Weights & Config-Driven Storage:** The legacy `models_data/` will be deprecated. Storage behavior will be dictated by a global `config.yaml` (or `.env`), allowing users to toggle between using the global `huggingface_hub` cache or keeping weights localized inside `app/engines/<plugin_name>/weights/`.
- **Version Control:** All weight directories and `custom_voices/` will be strictly excluded from the GitHub repository via `.gitignore`. This keeps the repository lightweight without needing to hide directories as dotfiles in the root.

### 3. Decentralized Capabilities & UI Defaults
The central UI (`gradio_app.py`) must be completely agnostic to the specific needs of a TTS engine.
Each `TTSPlugin` subclass must define its own:
- `display_name` (e.g., "Kokoro v0.19 (ONNX)")
- Requirements (e.g., `requires_cloning_transcript = True`)
- Recommended Audio Reference Lengths (for the UI tooltips)
- **UI Parameters Defaults:** (Temperature, Top-K, Repetition Penalty, CFG, Exaggeration, etc.)

### 4. Decentralized Voice Cloning
Voice cloning logic is currently heavily centralized inside `clones.py`, containing hardcoded assumptions about how different models serialize data (`.wav`, `.txt`, `.base`, `.safetensors`).
- The generic `custom_voices/<engine_id>/` structure will remain.
- However, each `TTSPlugin` will be responsible for defining its own `save_clone()` and `get_clone_path()` implementations, serializing the data in whatever format the model uniquely requires.

## Anticipated Architecture

```text
app/
├── api/
│   └── ws.py                 <-- Agnostic WebSocket router
├── ui/
│   └── gradio_app.py         <-- Dynamic UI generator
└── engines/
    ├── manager.py            <-- Scans plugins & manages active state
    ├── interface.py          <-- Defines TTSPlugin abstract base class
    │
    ├── kokoro_onnx/
    │   ├── engine.py         <-- Implements TTSPlugin for Kokoro
    │   └── ...
    │
    └── chatterbox_onnx/
        ├── engine.py         <-- Implements TTSPlugin for Chatterbox
        └── ...
```

This strategy ensures high maintainability, rapid integration of upcoming state-of-the-art models, and a robust open-source contribution framework.
