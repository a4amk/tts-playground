The **Universal TTS Control Center** is built for developer speed. Adding a new model is a two-step process that requires zero changes to the UI or the API routing itself. Engines follow a **Model (Assets)** and **Runtime (Inference)** separation pattern.

---

## Step 1: Create the Engine Folder

Create a folder: `app/engines/my_engine/`.

## Step 2: Implement `model.py` and `runtime.py`

### `app/engines/my_engine/model.py`
```python
class MyModel:
    def __init__(self):
        # Load weights here
        self.pipeline = "ready"
    def get_voices(self):
        return ["v1", "v2"]

my_model = MyModel()
```

### `app/engines/my_engine/runtime.py`
```python
import numpy as np
from typing import Generator, Tuple, Optional, List
from ..base import BaseTTS
from .model import my_model  # Link back to the loader

class MyRuntime(BaseTTS):
    def get_available_voices(self) -> List[str]:
        return my_model.get_voices()

    def generate_stream(self, text: str, voice: str, speed: float, **kwargs) -> Generator[np.ndarray, None, None]:
        # Perform inference using my_model.pipeline
        yield chunk

    def generate_batch(self, text: str, voice: str, speed: float, **kwargs) -> Optional[Tuple[int, np.ndarray]]:
        return (24000, final_audio_int16)

my_runtime = MyRuntime()
```

## Step 3: Register the Model

Open `app/engines/registry.py` and add your engine to the `models` dictionary:

```python
from .kokoro.runtime import kokoro_runtime
from .my_engine.runtime import my_runtime  # Import your new runtime

models = {
    "kokoro": kokoro_runtime,
    "mycustom": my_runtime  # Register it here
}
```

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

## 🛠️ Case Study: Adding an ONNX Model

If you have a model in ONNX format (e.g., Piper or a converted Kokoro), you would follow the same pattern:

1.  **Engine Folder**: `app/engines/onnx_piper/`
2.  **`model.py`**:
    ```python
    import onnxruntime as ort
    class PiperModel:
        def __init__(self):
            # Load the onnx session once
            self.session = ort.InferenceSession("model.onnx")
    piper_model = PiperModel()
    ```
3.  **`runtime.py`**:
    ```python
    from .model import piper_model
    class PiperRuntime(BaseTTS):
        def generate_stream(self, text, voice, speed, **kwargs):
            # 1. Convert text to input tensors
            # 2. run piper_model.session.run(...)
            # 3. yield audio_chunk
            yield chunk
    piper_runtime = PiperRuntime()
    ```
4.  **Register**: In `registry.py`, add `"piper": piper_runtime`.

---

## ✨ Automated UI Support

Once registered:
1.  **Engine Dropdown**: The Gradio UI will automatically detect the new key (`mycustom`) and add it to the "TTS Engine" dropdown.
2.  **Voice Population**: Selecting "mycustom" in the UI will trigger a reactive update that calls your engine's `get_available_voices()` method, populating the voice dropdown instantly.
3.  **WebSocket Support**: Sending `{"model": "mycustom"}` to the WebSocket endpoint will automatically route to your engine.

---

## 💡 Pro Tips for Performance

*   **Yielding Chunks**: Try to yield audio as soon as a single sentence or even a single word is ready (if the model supports it). This keeps **TTFB** ultra-low.
*   **Audio Depth**: Ensure you are yielding **`Float32`** numpy arrays for the `generate_stream` method. The browser's `AudioContext` expects this bit depth.
*   **Sample Rate**: The current frontend is hardcoded for **24,000Hz**. If your model uses another rate (e.g., 44,100Hz), you may need to upsample/downsample in the `generate` generator or update the frontend's `AudioContext` initialization in `app/ui/js_snippets.py`.
