import os
import torch
import subprocess
import numpy as np
import asyncio
import scipy.signal
import logging
import librosa
from typing import AsyncGenerator, Tuple, Optional, List, Dict, Any
from pocket_tts.models.tts_model import TTSModel, _import_model_state
from pocket_tts.default_parameters import DEFAULT_VARIANT
from ..interface import TTSPlugin

logger = logging.getLogger(__name__)

class PocketTTSEngine(TTSPlugin):
    """
    Standardized Pocket-TTS Engine.
    """
    def __init__(self):
        self._id = "pocket_tts"
        self._display_name = "Pocket-TTS (Kyutai/Mimi)"
        
        self._model = None
        self._cached_prompt_states = {}
        
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        self.base_dir = os.path.join(project_root, "models_data/pocket-tts")
        self.local_config = os.path.join(self.base_dir, "b6369a24.yaml")
        self.voice_dir = os.path.join(self.base_dir, "voices")
        self.clones_dir = os.path.join(project_root, "custom_voices/pocket_tts")
        os.makedirs(self.clones_dir, exist_ok=True)

    @property
    def id(self) -> str:
        return self._id

    @property
    def display_name(self) -> str:
        return self._display_name

    def get_standard_controls(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "speed", "label": "Playback Speed",
                "info": "Adjusts the relative tempo. Works with stream and batch modes.",
                "min": 0.5, "max": 2.0, "step": 0.1, "default": 1.0
            },
            {
                "id": "temp", "label": "Sampling temperature",
                "info": "Controls randomness. High = creative, Low = precise. Recommended: 0.7. Works with stream and batch.",
                "min": 0.1, "max": 2.0, "step": 0.1, "default": 0.7
            },
            {
                "id": "seed", "label": "Random Seed",
                "info": "Set to 0 for random results, or a fixed number for reproducibility. Works with stream and batch.",
                "min": 0, "max": 999999, "step": 1, "default": 0
            }
        ]

    def get_ui_config(self) -> Dict[str, Any]:
        return {
            "temp": 0.7,
            "seed": 0,
            "speed": 1.0
        }

    def get_cloning_config(self) -> Dict[str, Any]:
        return {
            "requires_cloning": True,
            "requires_transcript": False,
            "instruction": "Upload a short clear audio sample. Mimi will extract acoustic features automatically. Use the 'Edit' (scissors) button to trim if the audio exceeds 15s."
        }

    def get_variants(self) -> List[Dict[str, Any]]:
        return [{"id": "pytorch", "label": "Mimi (PyTorch FP32)", "default": True}]

    def get_available_voices(self) -> List[str]:
        predefined = []
        if os.path.exists(self.voice_dir):
            predefined = [f.replace(".safetensors", "") for f in os.listdir(self.voice_dir) if f.endswith(".safetensors")]
        
        user_clones = self.list_clones()
        return sorted(list(set(predefined + user_clones)))

    def get_available_languages(self) -> List[str]:
        return ["en"]

    def is_installed(self) -> bool:
        try:
            import pocket_tts
            return True
        except ImportError:
            return False

    def install_dependencies(self):
        logger.info(f"Installing dependencies for {self.id}...")
        subprocess.run(["pip", "install", "pocket-tts", "torch", "scipy"], check=False)

    def load(self, variant: Optional[str] = None):
        if self._model is None:
            from ...config import get_device
            device = get_device("cpu")
            config_path = self.local_config if os.path.exists(self.local_config) else DEFAULT_VARIANT
            logger.info(f"Loading Pocket-TTS model from {config_path} on {device}...")
            self._model = TTSModel.load_model(config_path)
            self._model.eval()
            self._model.to(device)

    def _get_prompt_state(self, voice_key: str):
        self.load()
        from ...config import get_device
        device = get_device("cpu")
        if voice_key not in self._cached_prompt_states:
             # 1. Check user clones
             clone_path = os.path.join(self.clones_dir, f"{voice_key}.safetensors")
             if not os.path.exists(clone_path):
                 clone_path = os.path.join(self.clones_dir, f"{voice_key}.pt")
             
             # 2. Check predefined voices
             if not os.path.exists(clone_path):
                 clone_path = os.path.join(self.voice_dir, f"{voice_key}.safetensors")
             
             if os.path.exists(clone_path):
                 if clone_path.endswith(".pt"):
                     self._cached_prompt_states[voice_key] = torch.load(clone_path, map_location=device, weights_only=True)
                 else:
                     self._cached_prompt_states[voice_key] = _import_model_state(clone_path)
             else:
                 # 3. Fallback: try to extract from a raw file if path provided (or default)
                 logger.info(f"Extracting state for voice: {voice_key}")
                 self._cached_prompt_states[voice_key] = self._model.get_state_for_audio_prompt(voice_key)
                 
        return self._cached_prompt_states[voice_key]

    async def generate_stream(self, text: str, voice: str, speed: float, variant: Optional[str] = None, **kwargs) -> AsyncGenerator[np.ndarray, None]:
        self.load()
        prompt_state = self._get_prompt_state(voice)
        
        # Temp handling
        self._model.temp = kwargs.get("temp", 1.0 / max(0.1, speed))
        
        seed = kwargs.get("seed", 0)
        if seed > 0:
            torch.manual_seed(seed)
        
        split_choice = kwargs.get("split_choice", "Sentences (Punctuation)")
        custom_regex = kwargs.get("custom_regex", "")
        chunks = self.split_text(text, split_choice, custom_regex)
        
        source_sr = self._model.config.mimi.sample_rate
        target_sr = 24000
        
        for chunk_text in chunks:
            if not chunk_text.strip(): continue
            
            # Pocket-TTS generate_audio_stream is a generator. 
            # We run it in a thread to keep the async loop happy.
            import queue
            import threading
            q = queue.Queue()
            
            def worker():
                try:
                    for chunk in self._model.generate_audio_stream(model_state=prompt_state, text_to_generate=chunk_text, copy_state=True):
                         audio = chunk.detach().cpu().numpy().flatten().astype(np.float32)
                     # skip per-chunk speed stretch to prevent artifacts
                         q.put(("data", audio))
                except Exception as e:
                    q.put(("error", e))
                finally:
                    q.put(("done", None))

            thread = threading.Thread(target=worker)
            thread.start()
            
            while True:
                msg_type, data = await asyncio.get_event_loop().run_in_executor(None, q.get)
                if msg_type == "done": break
                if msg_type == "error": 
                    logger.error(f"Pocket-TTS stream error: {data}")
                    break
                
                audio_np = data
                if source_sr != target_sr:
                    num_samples = int(len(audio_np) * target_sr / source_sr)
                    audio_np = scipy.signal.resample(audio_np, num_samples)
                yield audio_np.astype(np.float32)
            
            thread.join()

    def generate_batch(self, text: str, voice: str, speed: float, variant: Optional[str] = None, **kwargs) -> Optional[Tuple[int, np.ndarray]]:
        self.load()
        prompt_state = self._get_prompt_state(voice)
        source_sr = self._model.config.mimi.sample_rate
        
        audio_chunks = []
        for chunk in self._model.generate_audio_stream(model_state=prompt_state, text_to_generate=text, copy_state=True):
            chunk_np = chunk.detach().cpu().numpy().flatten().astype(np.float32)
            # skip per-chunk speed stretch
            audio_chunks.append(chunk_np)
        
        if not audio_chunks:
             return None
             
        full_audio = np.concatenate(audio_chunks)
        target_sr = 24000
        if source_sr != target_sr:
            num_samples = int(len(full_audio) * target_sr / source_sr)
            full_audio = scipy.signal.resample(full_audio, num_samples)
        
        if speed != 1.0:
            import librosa
            full_audio = librosa.effects.time_stretch(full_audio, rate=speed)
            
        return target_sr, (full_audio * 32767).astype(np.int16)

    def save_clone(self, name: str, audio_path: str, transcript: Optional[str] = None):
        try:
            import librosa
            duration = librosa.get_duration(path=audio_path)
            if duration > 15.0:
                raise ValueError(f"Audio is {duration:.1f}s long. Please use the 'Edit' (scissors) button to trim it under 15s.")
        except ImportError:
            pass
        self.load()
        target_path = os.path.join(self.clones_dir, f"{name}.safetensors")
        # Extract features
        state = self._model.get_state_for_audio_prompt(audio_path, truncate=True)
        # Save state (Safetensors is standard for Kyutai)
        from safetensors.torch import save_file
        save_file(state, target_path)
        logger.info(f"Saved Pocket-TTS clone: {name} to {target_path}")

    def list_clones(self) -> List[str]:
        if not os.path.exists(self.clones_dir):
            return []
        return sorted([f.split(".safetensors")[0].split(".pt")[0] for f in os.listdir(self.clones_dir) if f.endswith(".safetensors") or f.endswith(".pt")])
