import os
import numpy as np
import scipy.signal
import asyncio
import logging
from typing import AsyncGenerator, Tuple, Optional, List, Dict, Any
from piper.voice import PiperVoice
from piper.config import SynthesisConfig
from ..interface import TTSPlugin

logger = logging.getLogger(__name__)

class PiperEngine(TTSPlugin):
    """
    Standardized Piper Engine.
    """
    def __init__(self):
        self._id = "piper"
        self._display_name = "Piper (VITS)"
        
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        self.voices_base = os.path.join(project_root, "models_data/piper-onnx")
        self._loaded_voices = {}

    @property
    def id(self) -> str:
        return self._id

    @property
    def display_name(self) -> str:
        return self._display_name

    def get_ui_config(self) -> Dict[str, Any]:
        return {
            "speed": 1.0,
            "noise_scale": 0.667,
            "noise_w_scale": 0.8
        }

    def get_cloning_config(self) -> Dict[str, Any]:
        return {
            "requires_cloning": False
        }

    def get_available_voices(self) -> List[str]:
        if not os.path.exists(self.voices_base):
             return []
        voices = []
        for d in os.listdir(self.voices_base):
             voice_dir = os.path.join(self.voices_base, d)
             if os.path.isdir(voice_dir):
                  if any(f.endswith(".onnx") for f in os.listdir(voice_dir)):
                       voices.append(d)
        return sorted(voices)

    def get_available_languages(self) -> List[str]:
        # Piper languages are usually embedded in the voice name/path
        return ["en-us", "auto"]

    def is_installed(self) -> bool:
        try:
            import piper
            return True
        except ImportError:
            return False

    def install_dependencies(self):
        logger.info(f"Installing dependencies for {self.id}...")
        os.system("pip install piper-tts")

    def load(self):
        """Lazy loading is handled per voice in _get_voice_pipeline."""
        pass

    def _get_voice_pipeline(self, voice_key: str):
        if voice_key not in self._loaded_voices:
            voice_dir = os.path.join(self.voices_base, voice_key)
            if not os.path.exists(voice_dir):
                 return None
                 
            onnx_files = [f for f in os.listdir(voice_dir) if f.endswith(".onnx")]
            if not onnx_files:
                 return None
            
            model_path = os.path.join(voice_dir, onnx_files[0])
            config_path = model_path + ".json"
            
            if not os.path.exists(config_path):
                 json_files = [f for f in os.listdir(voice_dir) if f.endswith(".json")]
                 if json_files:
                      config_path = os.path.join(voice_dir, json_files[0])
            
            logger.info(f"Loading Piper voice from {model_path}...")
            self._loaded_voices[voice_key] = PiperVoice.load(model_path, config_path=config_path)
            
        return self._loaded_voices[voice_key]

    async def generate_stream(self, text: str, voice: str, speed: float, **kwargs) -> AsyncGenerator[np.ndarray, None]:
        ls = 1.0 / speed if speed > 0 else 1.0
        ns = kwargs.get("noise_scale", 0.667)
        nw = kwargs.get("noise_w_scale", 0.8)
        
        voice_pipeline = self._get_voice_pipeline(voice)
        if not voice_pipeline:
             logger.error(f"Error: Could not load voice '{voice}' for Piper!")
             return
             
        target_sr = 24000
        source_sr = voice_pipeline.config.sample_rate
        
        syn_config = SynthesisConfig(length_scale=ls, noise_scale=ns, noise_w_scale=nw)
        
        split_choice = kwargs.get("split_choice", "Both (Newlines & Sentences)")
        custom_regex = kwargs.get("custom_regex", r'\n+')
        chunks = self.split_text(text, split_choice, custom_regex)
        
        for chunk_text in chunks:
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
        ns = kwargs.get("noise_scale", 0.667)
        nw = kwargs.get("noise_w_scale", 0.8)
        
        voice_pipeline = self._get_voice_pipeline(voice)
        if not voice_pipeline:
             return None
             
        all_chunks = []
        syn_config = SynthesisConfig(length_scale=ls, noise_scale=ns, noise_w_scale=nw)
        for chunk in voice_pipeline.synthesize(text, syn_config=syn_config):
             all_chunks.append(chunk.audio_int16_array)
             
        if not all_chunks:
             return None
        
        all_audio = np.concatenate(all_chunks)
        return (voice_pipeline.config.sample_rate, all_audio.astype(np.int16))

    def save_clone(self, name: str, audio_path: str, transcript: Optional[str] = None):
        raise NotImplementedError("Piper does not support zero-shot cloning.")

    def list_clones(self) -> List[str]:
        return []
