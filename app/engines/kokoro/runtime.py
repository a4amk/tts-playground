import numpy as np
from typing import AsyncGenerator, Tuple, Optional, List
from ..base import BaseTTS
from .model import kokoro_model

class KokoroRuntime(BaseTTS):
    """
    Handles Kokoro inference (The 'Runtime' logic).
    Separates executing inference from loading the weights.
    """
    def get_available_voices(self) -> List[str]:
        # Delegate voice check to model (Assets/Loader part)
        return kokoro_model.get_voices()

    def get_available_languages(self) -> List[str]:
        return ["a", "b", "en", "es", "fr", "hi", "it", "ja", "ko", "pt", "zh", "auto"]
    async def generate_stream(self, text: str, voice: str, speed: float, **kwargs) -> AsyncGenerator[np.ndarray, None]:
        lang = kwargs.get("lang", "a")
        split_choice = kwargs.get("split_choice", "Both (Newlines & Sentences)")
        custom_regex = kwargs.get("custom_regex", r'\n+')
        
        # Access the runtime pipeline from the model loader
        pipeline = kokoro_model.get_pipeline(lang)
        chunks = self.split_text(text, split_choice, custom_regex)
        
        for chunk_text in chunks:
            generator = pipeline(chunk_text.strip(), voice=voice, speed=speed, split_pattern=None)
        
        # Wrap sync generator into an async one.
        for gs, ps, audio in generator:
            if hasattr(audio, 'numpy'):
                chunk = audio.numpy()
            else:
                chunk = audio
                
            if chunk is not None and len(chunk) > 0:
                yield chunk

    def generate_batch(self, text: str, voice: str, speed: float, **kwargs) -> Optional[Tuple[int, np.ndarray]]:
        if not text.strip():
            return None
            
        import asyncio
        # To call an async generator in a sync context (generate_batch is sync as per BaseTTS)
        async def run_stream():
            chunks = []
            async for chunk in self.generate_stream(text, voice, speed, **kwargs):
                 chunks.append(chunk)
            return chunks

        # In case we aren't in a loop (Gradio runs its own)
        try:
             loop = asyncio.get_event_loop()
             if loop.is_running():
                  # This happens in Gradio. We can't use run() here.
                  # But generate_batch is called from a thread usually.
                  # Or we use our own small loop.
                  import nest_asyncio
                  nest_asyncio.apply()
                  chunks = asyncio.run(run_stream())
             else:
                  chunks = asyncio.run(run_stream())
        except Exception:
             # Fallback: manually Iterate
             # Wait, if we are in a thread we can just create a new loop.
             new_loop = asyncio.new_event_loop()
             chunks = new_loop.run_until_complete(run_stream())
             new_loop.close()

        if not chunks:
            return None
            
        final_audio = np.concatenate(chunks)
        final_audio_int16 = (final_audio * 32767).astype(np.int16)
        return (24000, final_audio_int16)

# Global singleton or per-instance
kokoro_runtime = KokoroRuntime()
