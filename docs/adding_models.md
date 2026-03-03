# Adding New Models

The **Universal TTS Control Center** is built for developer speed. Adding a new model is a two-step process that requires zero changes to the UI or the API routing itself. Engines follow a **Model (Assets)** and **Runtime (Inference)** separation pattern.

---

## Step 1: Create the Engine Folder

Create a folder: `app/engines/my_engine/`.

## Step 2: Implement `engine.py`

Create `app/engines/my_engine/engine.py`. Your class must inherit from `TTSPlugin` and implement the abstract methods.

```python
import numpy as np
from typing import AsyncGenerator, Tuple, Optional, List, Dict, Any
from ..interface import TTSPlugin

class MyEngine(TTSPlugin):
    def __init__(self):
        self._id = "mycustom"
        self._display_name = "My Custom Model"
        # Load weights or sessions here
    
    @property
    def id(self) -> str: return self._id
    
    @property
    def display_name(self) -> str: return self._display_name

    def get_standard_controls(self) -> List[Dict[str, Any]]:
        return [{"id": "speed", "label": "Speed", "default": 1.0}]

    def get_variants(self) -> List[Dict[str, Any]]:
        return [{"id": "fp32", "label": "FP32", "default": True}]

    def get_cloning_config(self) -> Dict[str, Any]:
        return {"requires_cloning": False}

    async def generate_stream(self, text, voice, speed, **kwargs):
        # Your inference logic here
        yield audio_chunk_float32

    def generate_batch(self, text, voice, speed, **kwargs):
        return (24000, audio_int16)
    
    # ... implement other abstract methods from interface.py
```

## Step 3: Auto-Discovery

You no longer need to manually register models! The `PluginManager` will automatically find your engine class if it's in the `app/engines/` directory and inherits from `TTSPlugin`. Just restart the server, and it will appear in the UI.

## Step 4: Add a Setup Script (Recommended)

To ensure your engine is portable, add a setup script in `scripts/setup_my_engine.sh`:
```bash
#!/bin/bash
# Install dependencies
venv/bin/pip install my-new-model-lib

# Download assets to the unified storage
mkdir -p models_data/my-engine
wget -P models_data/my-engine/ https://example.com/weights.bin
```

---

## Case Study: Adding an ONNX Model

If you have a model in ONNX format (e.g., Piper or a converted Kokoro), you would follow the same pattern:

1.  **Engine Folder**: `app/engines/onnx_piper/`
2.  **`engine.py`**:
    ```python
    import onnxruntime as ort
    from ..interface import TTSPlugin

    class PiperEngine(TTSPlugin):
        def __init__(self):
            self._id = "piper_onnx"
            self._display_name = "Piper ONNX"
            # Load the onnx session once
            self.session = ort.InferenceSession("models_data/piper/model.onnx")

        async def generate_stream(self, text, voice, speed, **kwargs):
            # 1. Convert text to input tensors
            # 2. run self.session.run(...)
            # 3. yield audio_chunk as float32 numpy array
            yield chunk
    ```

---

## Automated UI Support

Once registered:
1.  **Engine Dropdown**: The Gradio UI will automatically detect the new key (`mycustom`) and add it to the "TTS Engine" dropdown.
2.  **Voice Population**: Selecting "mycustom" in the UI will trigger a reactive update that calls your engine's `get_available_voices()` method, populating the voice dropdown instantly.
3.  **WebSocket Support**: Sending `{"model": "mycustom"}` to the WebSocket endpoint will automatically route to your engine.

---

## Pro Tips for Performance

*   **Yielding Chunks**: Try to yield audio as soon as a single sentence or even a single word is ready (if the model supports it). This keeps **TTFA** ultra-low.
*   **Audio Depth**: Ensure you are yielding **`Float32`** numpy arrays for the `generate_stream` method. The browser's `AudioContext` expects this bit depth.
*   **Sample Rate**: The current frontend is hardcoded for **24,000Hz**. If your model uses another rate (e.g., 44,100Hz), you may need to upsample/downsample in the `generate` generator or update the frontend's `AudioContext` initialization in `app/ui/js_snippets.py`.
