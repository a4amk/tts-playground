from .kokoro.runtime import kokoro_runtime
from .kokoro_onnx.runtime import kokoro_onnx_runtime
from .piper.runtime import piper_runtime
from .pocket_tts.runtime import pocket_tts_runtime
from .zipvoice.runtime import zipvoice_runtime
from .genie.runtime import genie_runtime
from .chatterbox_onnx.runtime import ChatterboxONNXRuntime
from .base import BaseTTS
from typing import Dict

# A basic registry dict so we can address models generically
models: Dict[str, BaseTTS] = {
    "kokoro": kokoro_runtime,
    "kokoro-onnx": kokoro_onnx_runtime,
    "piper": piper_runtime,
    "pocket-tts": pocket_tts_runtime,
    "zipvoice": zipvoice_runtime,
    "genie": genie_runtime,
    "chatterbox-turbo-onnx": ChatterboxONNXRuntime()
}
