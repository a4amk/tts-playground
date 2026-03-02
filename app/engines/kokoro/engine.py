```python
import os
import numpy as np
import logging
import librosa
import subprocess
import asyncio
from typing import AsyncGenerator, Tuple, Optional, List, Dict, Any
from kokoro import KPipeline # Keep this, as KPipeline is used
from ..interface import TTSPlugin
from ...config import get_device

logger = logging.getLogger(__name__)

class KokoroPyTorchEngine(TTSPlugin):
    """
    Standardized Kokoro PyTorch Engine.
    """
    def __init__(self):
        # Metadata
        self._id = "kokoro"
        self._display_name = "Kokoro v0.19 (PyTorch)"
        
        # Internal state
        self.pipelines = {}
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        self.voice_dir = os.path.join(project_root, "models_data/kokoro-82m/voices")
        self.device = get_device("cpu")
        
        # Ensure espeak-ng data path (specific to some environments)
        if os.path.exists("/usr/lib/aarch64-linux-gnu/espeak-ng-data"):
            os.environ["ESPEAK_DATA_PATH"] = "/usr/lib/aarch64-linux-gnu/espeak-ng-data"
        elif os.path.exists("/usr/lib/x86_64-linux-gnu/espeak-ng-data"):
             os.environ["ESPEAK_DATA_PATH"] = "/usr/lib/x86_64-linux-gnu/espeak-ng-data"

    @property
    def id(self) -> str:
        return self._id

    @property
    def display_name(self) -> str:
        return self._display_name

    def get_standard_controls(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "speed",
                "label": "Synthesis Speed",
                "info": "Controls the reading speed of the model. 1.0 is default. Works with both streaming and batch modes.",
                "min": 0.5, "max": 2.0, "step": 0.1, "default": 1.0
            }
        ]

    def get_ui_config(self) -> Dict[str, Any]:
        return {
            "speed": 1.0,
            "lang": "en-us"
        }

    def get_variants(self) -> List[Dict[str, Any]]:
        return [{"id": "pytorch", "label": "PyTorch (FP32)", "default": True}]

    def get_cloning_config(self) -> Dict[str, Any]:
        return {
            "requires_cloning": False
        }

    def get_available_voices(self) -> List[str]:
        available_voices = ["af_heart"]
        if os.path.exists(self.voice_dir):
            voices = sorted([f.split('.')[0] for f in os.listdir(self.voice_dir) if f.endswith('.pt')])
            if voices:
                available_voices = voices
        return available_voices

    def get_available_languages(self) -> List[str]:
        return ["a", "b", "en", "es", "fr", "hi", "it", "ja", "ko", "pt", "zh", "auto"]

    def is_installed(self) -> bool:
        try:
            import kokoro
            return True
        except ImportError:
            return False

    def install_dependencies(self):
        # Simplified for now: just log or call pip
        logger.info(f"Installing dependencies for {self.id}...")
        subprocess.run(["pip", "install", "kokoro"], check=False)

    def load(self, variant: Optional[str] = None):
        """Lazy loading is handled within generate_stream via get_pipeline."""
        pass

    def _get_pipeline(self, lang_code: str):
        if lang_code not in self.pipelines:
            logger.info(f"Loading KPipeline for lang_code '{lang_code}'...")
            try:
                self.pipelines[lang_code] = KPipeline(lang_code=lang_code)
                self.pipelines[lang_code].model.to(self.device)
                logger.info(f"Loaded Kokoro pipeline for language '{lang_code}' on {self.device}")
            except Exception as e:
                if lang_code != 'a':
                    return self._get_pipeline('a')
                raise e
        return self.pipelines[lang_code]

    async def generate_stream(self, text: str, voice: str, speed: float, variant: Optional[str] = None, **kwargs) -> AsyncGenerator[np.ndarray, None]:
        # Filter out named arguments if they slipped into kwargs
        kwargs.pop("text", None)
        kwargs.pop("voice", None)
        kwargs.pop("speed", None)
        lang = kwargs.get("lang", "a")
        split_choice = kwargs.get("split_choice", "Both (Newlines & Sentences)")
        custom_regex = kwargs.get("custom_regex", r'\n+')
        
        pipeline = self._get_pipeline(lang)
        chunks = self.split_text(text, split_choice, custom_regex)
        
        for chunk_text in chunks:
            # KPipeline generator is synchronous
            generator = pipeline(chunk_text.strip(), voice=voice, speed=1.0, split_pattern=None)
            for gs, ps, audio in generator:
                if hasattr(audio, 'numpy'):
                    chunk = audio.numpy()
                else:
                    chunk = audio
                    
                if chunk is not None and len(chunk) > 0:
                    if speed != 1.0:
                        chunk = librosa.effects.time_stretch(chunk, rate=speed)
                    yield chunk

    def generate_batch(self, text: str, voice: str, speed: float, variant: Optional[str] = None, **kwargs) -> Optional[Tuple[int, np.ndarray]]:
        if not text.strip():
            return None
            
        async def run_stream():
            chunks = []
            async for chunk in self.generate_stream(text, voice, speed, variant=variant, **kwargs):
                 chunks.append(chunk)
            return chunks

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                chunks = asyncio.run(run_stream())
            else:
                chunks = asyncio.run(run_stream())
        except Exception:
            new_loop = asyncio.new_event_loop()
            chunks = new_loop.run_until_complete(run_stream())
            new_loop.close()

        if not chunks:
            return None
            
        final_audio = np.concatenate(chunks)
        final_audio_int16 = (final_audio * 32767).astype(np.int16)
        return (24000, final_audio_int16)

    def save_clone(self, name: str, audio_path: str, transcript: Optional[str] = None, **kwargs):
        raise NotImplementedError("Kokoro PyTorch does not support zero-shot cloning in this implementation.")

    def list_clones(self) -> List[str]:
        return []
