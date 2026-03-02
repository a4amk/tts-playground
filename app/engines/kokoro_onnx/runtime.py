import numpy as np
import re
from typing import AsyncGenerator, Tuple, Optional, List
from ..base import BaseTTS
from .model import kokoro_onnx_model

class KokoroONNXRuntime(BaseTTS):
    def get_available_voices(self) -> List[str]:
        return kokoro_onnx_model.get_voices()

    def get_available_languages(self) -> List[str]:
        return ["en-us", "en-gb", "fr-fr", "ja", "ko", "zh", "auto"]


    async def generate_stream(self, text: str, voice: str, speed: float, **kwargs) -> AsyncGenerator[np.ndarray, None]:
        split_choice = kwargs.get("split_choice", "Both (Newlines & Sentences)")
        custom_regex = kwargs.get("custom_regex", r'\n+')
        segments = self.split_text(text.strip(), split_choice, custom_regex)
        
        kokoro = kokoro_onnx_model.kokoro
        
        for segment in segments:
            # create_stream is an async generator in the library
            try:
                async for samples, _ in kokoro.create_stream(segment, voice=voice, speed=speed):
                    if samples is not None and len(samples) > 0:
                        yield samples
            except Exception as e:
                # Fallback to standard batch if something goes wrong in the async generator
                print(f"ONNX Stream error: {e}, falling back to creation.")
                samples, _ = kokoro.create(segment, voice=voice, speed=speed)
                if samples is not None and len(samples) > 0:
                    yield samples

    def generate_batch(self, text: str, voice: str, speed: float, **kwargs) -> Optional[Tuple[int, np.ndarray]]:
        if not text.strip():
            return None
        
        samples, rate = kokoro_onnx_model.kokoro.create(text, voice=voice, speed=speed)
        
        if samples is None:
            return None
            
        final_audio_int16 = (samples * 32767).astype(np.int16)
        return (rate, final_audio_int16)

# Global singleton or per-instance
kokoro_onnx_runtime = KokoroONNXRuntime()
