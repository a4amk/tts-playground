import numpy as np
import scipy.signal
import asyncio
from typing import AsyncGenerator, Tuple, Optional, List
from ..base import BaseTTS
from .model import piper_model
from piper.config import SynthesisConfig

class PiperRuntime(BaseTTS):
    def get_available_voices(self) -> List[str]:
        return piper_model.get_available_voices()

    async def generate_stream(self, text: str, voice: str, speed: float, **kwargs) -> AsyncGenerator[np.ndarray, None]:
        ls = 1.0 / speed if speed > 0 else 1.0
        
        voice_pipeline = piper_model.get_voice_pipeline(voice)
        if not voice_pipeline:
             print(f"Error: Could not load voice '{voice}' for Piper!")
             return
             
        target_sr = 24000
        source_sr = voice_pipeline.config.sample_rate # 22050 usually
        
        syn_config = SynthesisConfig(length_scale=ls)
        
        split_choice = kwargs.get("split_choice", "Both (Newlines & Sentences)")
        custom_regex = kwargs.get("custom_regex", r'\n+')
        chunks = self.split_text(text, split_choice, custom_regex)
        
        for chunk_text in chunks:
             # Feed one sentence/chunk at a time to ensure low-latency delivery per chunk
             synth_gen = voice_pipeline.synthesize(chunk_text, syn_config=syn_config)
             for chunk in synth_gen:
                  if hasattr(chunk, 'audio_float_array'):
                       audio_float32 = chunk.audio_float_array
                  else:
                       audio_float32 = chunk.audio_int16_array.astype(np.float32) / 32767.0
                  
                  if source_sr != target_sr:
                      num_samples = int(len(audio_float32) * target_sr / source_sr)
                      audio_float32 = scipy.signal.resample(audio_float32, num_samples)
                  
                  yield audio_float32.astype(np.float32)
                  await asyncio.sleep(0.001)

    def generate_batch(self, text: str, voice: str, speed: float, **kwargs) -> Optional[Tuple[int, np.ndarray]]:
        ls = 1.0 / speed if speed > 0 else 1.0
        voice_pipeline = piper_model.get_voice_pipeline(voice)
        if not voice_pipeline:
             return None
             
        all_chunks = []
        syn_config = SynthesisConfig(length_scale=ls)
        for chunk in voice_pipeline.synthesize(text, syn_config=syn_config):
             all_chunks.append(chunk.audio_int16_array)
             
        if not all_chunks:
             return None
        
        all_audio = np.concatenate(all_chunks)
        return (voice_pipeline.config.sample_rate, all_audio.astype(np.int16))

piper_runtime = PiperRuntime()
