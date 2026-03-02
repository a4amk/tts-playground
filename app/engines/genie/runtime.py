import numpy as np
import asyncio
from typing import AsyncGenerator, Tuple, Optional, List
from ..base import BaseTTS
from .model import genie_model, GENIE_AVAILABLE

class GenieRuntime(BaseTTS):
    def get_available_voices(self) -> List[str]:
        if not GENIE_AVAILABLE:
            return []
        return genie_model.get_available_voices()

    def get_available_languages(self) -> List[str]:
        return ["auto", "Japanese", "English", "Chinese", "Korean", "Hybrid-Chinese-English"]
    async def generate_stream(self, text: str, voice: str, speed: float, **kwargs) -> AsyncGenerator[np.ndarray, None]:
        if not GENIE_AVAILABLE:
            yield np.zeros(0, dtype=np.float32)
            return

        # Ensure character is loaded
        from ...engines.clones import clones_manager
        
        is_clone = voice.endswith('.wav')
        base_voice = voice
        clone_audio = None
        clone_text = None

        if is_clone:
            clone_audio, clone_text, base_voice = clones_manager.get_genie_clone_data(voice)
        
        if not genie_model.load_character(base_voice):
            print(f"Genie: Failed to load character {base_voice}")
            return

        import genie_tts as genie
        
        if is_clone and clone_audio and clone_text:
            # Override reference audio for this base voice temporarily or semi-permanently
            lang = kwargs.get("lang", "auto")
            if lang == "auto":
                lang = "English" # Default or pull from CHARA_LANG dict in genie
            genie.set_reference_audio(base_voice, clone_audio, clone_text, lang)
        
        actual_voice = base_voice
        
        # genie.tts_async(character_name, text, play=False, split_sentence=False)
        # Note: speed is not directly supported in the simple tts() call of genie yet
        # but we can implement it by resampling or just ignoring for now if the engine doesn't support it.
        # Actually, GPT-SoVITS supports speed during inference by changing the latent duration.
        # Looking at genie's Inference.py, it takes a speed parameter in model.sample (or similar).
        # However, the top-level tts_async doesn't expose it.
        # For now, we'll use the default speed.
        
        try:
            async for chunk_bytes in genie.tts_async(
                character_name=actual_voice,
                text=text,
                play=False,
                split_sentence=True
            ):
                if chunk_bytes:
                    # Convert int16 bytes to float32 ndarray
                    # Genie uses 32000Hz (based on TTSPlayer.py)
                    audio_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
                    audio_float32 = audio_int16.astype(np.float32) / 32767.0
                    
                    # Our system uses 24000Hz as standard in js_snippets.py
                    # We should ideally resample here to 24000Hz
                    # using scipy.signal.resample or soxr
                    
                    try:
                        import soxr
                        resampled = soxr.resample(audio_float32, 32000, 24000)
                        yield resampled
                    except ImportError:
                        # Fallback to 32000 and hope frontend handles it or we update frontend
                        yield audio_float32
        except Exception as e:
            print(f"Genie streaming error: {e}")

    def generate_batch(self, text: str, voice: str, speed: float, **kwargs) -> Optional[Tuple[int, np.ndarray]]:
        # Sync version
        import asyncio
        loop = asyncio.new_event_loop()
        async def _run():
            chunks = []
            async for chunk in self.generate_stream(text, voice, speed, **kwargs):
                if len(chunk) > 0:
                    chunks.append(chunk)
            if chunks:
                return np.concatenate(chunks)
            return np.zeros(0, dtype=np.float32)
            
        full_audio = loop.run_until_complete(_run())
        loop.close()
        
        if len(full_audio) > 0:
            return (24000, full_audio)
        return None

genie_runtime = GenieRuntime()
