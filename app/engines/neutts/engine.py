import os
import torch
import numpy as np
import logging
import json
import asyncio
from typing import AsyncGenerator, Tuple, Optional, List, Dict, Any
from ..interface import TTSPlugin
from ...config import get_device
from ...utils import secure_path_join

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

class NeuTTSEngine(TTSPlugin):
    """
    NeuTTS Engine (Air & Nano)
    Supports zero-shot cloning, streaming, and multiple quantizations.
    """

    def __init__(self):
        super().__init__()
        self.model = None
        self.current_variant = None
        self.clones_dir = os.path.join(PROJECT_ROOT, "custom_voices", "neutts")
        os.makedirs(self.clones_dir, exist_ok=True)

    @property
    def id(self) -> str:
        return "neutts"

    @property
    def display_name(self) -> str:
        return "NeuTTS Air / Nano"

    @property
    def name(self) -> str:
        return self.display_name

    @property
    def description(self) -> str:
        return "High-performance TTS by Neuphonic with zero-shot cloning. Air version for quality, Nano for speed."

    def get_ui_config(self) -> Dict[str, Any]:
        return {
            "speed": 1.0,
            "temp": 1.0
        }

    def get_standard_controls(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "temp",
                "name": "Temperature",
                "label": "Temperature (Randomness)",
                "type": "slider",
                "min": 0.1,
                "max": 2.0,
                "default": 1.0,
                "step": 0.1,
                "info": "Controls the randomness of the synthesis. Lower is more stable, higher is more creative."
            },
            {
                "id": "speed",
                "name": "Speed",
                "label": "Relative Playback Speed",
                "type": "slider",
                "min": 0.5,
                "max": 2.0,
                "default": 1.0,
                "step": 0.1,
                "info": "Values < 1.0 are slower, > 1.0 are faster."
            }
        ]

    def get_extra_controls(self) -> List[Dict[str, Any]]:
        return []

    def get_cloning_config(self) -> Dict[str, Any]:
        return {
            "requires_cloning": True,
            "requires_transcript": True,
            "instruction": "Upload a 5-10s audio clip AND provide its exact transcript. If longer than 15s, use the 'Edit' (scissors) button on the audio uploader to trim it. NeuTTS uses this for zero-shot cloning."
        }

    def get_variants(self) -> List[Dict[str, Any]]:
        return [
            {"id": "air-q4-gguf", "label": "Air (4-bit GGUF) - High Quality, Fast", "default": True},
            {"id": "air-fp32-onnx", "label": "Air (FP32 ONNX)"},
            {"id": "nano-q4-gguf", "label": "Nano (4-bit GGUF) - Ultra Fast"},
            {"id": "nano-torch", "label": "Nano (PyTorch) - Standard"},
        ]

    def get_available_voices(self) -> List[str]:
        return self.list_voices()

    def get_available_languages(self) -> List[str]:
        return ["auto", "English", "Spanish", "German", "French"]

    def is_installed(self) -> bool:
        try:
            import neutts
            return True
        except ImportError:
            return False

    def install_dependencies(self):
        # Already handled by setup_neutts.sh, but we can trigger it if needed
        import subprocess
        subprocess.run(["bash", "scripts/setup_neutts.sh"], check=True)

    def _get_repo_for_variant(self, variant: str) -> str:
        mapping = {
            "air-q4-gguf": "neuphonic/neutts-air-q4-gguf",
            "air-fp32-onnx": "neuphonic/neutts-air-fp32-onnx",
            "nano-q4-gguf": "neuphonic/neutts-nano-q4-gguf",
            "nano-torch": "neuphonic/neutts-nano",
        }
        return mapping.get(variant, "neuphonic/neutts-air-q4-gguf")

    def load(self, variant: Optional[str] = "air-q4-gguf"):
        if self.model and self.current_variant == variant:
            return

        from neutts import NeuTTS
        
        device = "gpu" if get_device() == "cuda" else "cpu"
        repo = self._get_repo_for_variant(variant)
        
        logger.info(f"Loading NeuTTS {variant} from {repo}")
        
        self.model = NeuTTS(
            backbone_repo=repo,
            backbone_device=device,
            codec_repo="neuphonic/neucodec",
            codec_device="cpu"
        )
        self.current_variant = variant

    def save_clone(self, name: str, audio_path: str, transcript: Optional[str] = None, **kwargs) -> bool:
        if not self.model:
            self.load()
            
        if not transcript:
            raise ValueError("NeuTTS requires a transcript for cloning.")

        # Demand manual trimming > 15s
        try:
            import librosa
            duration = librosa.get_duration(path=audio_path)
            if duration > 15.0:
                raise ValueError(f"Audio is {duration:.1f}s long. Please use the 'Edit' (scissors) button on the audio uploader to trim it under 15s.")
        except ImportError:
            pass

        # encode_reference is actually synchronous in neutts library
        ref_codes = self.model.encode_reference(audio_path)
        
        clone_data = {
            "ref_text": transcript,
            "ref_codes": ref_codes.tolist() if hasattr(ref_codes, "tolist") else list(ref_codes)
        }
        
        safe_name = "".join(c for c in name if c.isalnum() or c in ("-", "_")).strip()
        dest_path = os.path.join(self.clones_dir, f"{safe_name}.json")
        
        with open(dest_path, "w") as f:
            json.dump(clone_data, f)
            
        return True

    def list_clones(self) -> List[str]:
        if not os.path.exists(self.clones_dir):
            return []
        return [f.replace(".json", "") for f in os.listdir(self.clones_dir) if f.endswith(".json")]

    def _load_clone_data(self, voice: str) -> Tuple[Optional[List[int]], str]:
        if not voice:
            return None, ""
            
        path = secure_path_join(self.clones_dir, f"{voice}.json")
        if not os.path.exists(path):
            return None, ""
            
        try:
            with open(path, "r") as f:
                data = json.load(f)
                return data.get("ref_codes"), data.get("ref_text", "")
        except Exception as e:
            logger.error(f"Failed to load clone {voice}: {e}")
            return None, ""

    def generate_batch(self, text: str, voice: str, speed: float = 1.0, variant: Optional[str] = None, **kwargs) -> Optional[Tuple[int, np.ndarray]]:
        if not self.model or self.current_variant != variant:
            self.load(variant=variant)

        ref_codes, ref_text = self._load_clone_data(voice)
        if ref_codes is None:
            return None

        # NeuTTS model.infer is sync, so we can call it directly in this sync method
        wav = self.model.infer(text, ref_codes, ref_text)
        
        if wav is None:
            return None

        # Post-process for speed if needed
        if speed != 1.0:
            import librosa
            wav = librosa.effects.time_stretch(wav, rate=speed)

        return self.model.sample_rate, wav

    async def generate_stream(self, text: str, voice: str, speed: float = 1.0, variant: Optional[str] = None, **kwargs) -> AsyncGenerator[np.ndarray, None]:
        if not self.model or self.current_variant != variant:
            await asyncio.to_thread(self.load, variant)

        ref_codes, ref_text = self._load_clone_data(voice)
        if ref_codes is None:
            return

        def _get_gen():
            return self.model.infer_stream(text, ref_codes, ref_text)

        gen = await asyncio.to_thread(_get_gen)
        
        while True:
            chunk = await asyncio.to_thread(next, gen, None)
            if chunk is None:
                break
            
            # Removed per-chunk time_stretch to avoid streaming artifacts (clicks/glitches)
            # if speed != 1.0:
            #     import librosa
            #     chunk = librosa.effects.time_stretch(chunk, rate=speed)
                
            yield chunk

    def list_voices(self) -> List[str]:
        # Voices are basically the clones
        # I'll return the clones list
        if not os.path.exists(self.clones_dir):
            logger.warning(f"NeuTTS clones dir does not exist: {self.clones_dir}")
            return []
        
        voices = [f.replace(".json", "") for f in os.listdir(self.clones_dir) if f.endswith(".json")]
        logger.info(f"NeuTTS found {len(voices)} clones in {self.clones_dir}: {voices}")
        return voices
