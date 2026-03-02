import os
import numpy as np
import asyncio
import logging
from typing import AsyncGenerator, Tuple, Optional, List, Dict, Any
from ..interface import TTSPlugin
from ...config import get_device

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
MODELS_BASE = os.path.join(PROJECT_ROOT, "models_data/genie")
GENIE_DATA_DIR = os.path.join(MODELS_BASE, "GenieData")
CHARACTERS_DIR = os.path.join(MODELS_BASE, "characters")

os.environ["GENIE_DATA_DIR"] = GENIE_DATA_DIR

class GenieEngine(TTSPlugin):
    """
    Standardized Genie TTS Engine.
    """
    def __init__(self):
        self._id = "genie"
        self._display_name = "Genie (High Fidelity GPT-SoVITS)"
        self._initialized = False
        self.device = get_device("cuda")
        
        self.base_dir = MODELS_BASE
        self.characters_dir = CHARACTERS_DIR
        self.genie_data_dir = GENIE_DATA_DIR
        self.clones_dir = os.path.join(PROJECT_ROOT, "custom_voices/genie")
        os.makedirs(self.clones_dir, exist_ok=True)

    @property
    def id(self) -> str:
        return self._id

    @property
    def display_name(self) -> str:
        return self._display_name

    def get_ui_config(self) -> Dict[str, Any]:
        return {
            "lang": "English"
        }

    def get_cloning_config(self) -> Dict[str, Any]:
        return {
            "requires_cloning": True,
            "requires_transcript": True,
            "instruction": "Upload a 5-10s clear audio sample AND provide the exact transcript for the sample."
        }

    def get_available_voices(self) -> List[str]:
        if not os.path.exists(self.characters_dir):
            return []
        
        voices = []
        for version in ["v2ProPlus", "v2"]:
            v_dir = os.path.join(self.characters_dir, "CharacterModels", version)
            if os.path.exists(v_dir):
                for chara in os.listdir(v_dir):
                    if os.path.isdir(os.path.join(v_dir, chara)):
                        voices.append(chara)
        
        clones = self.list_clones()
        return sorted(list(set(voices + clones)))

    def get_available_languages(self) -> List[str]:
        return ["Japanese", "English", "Chinese", "Korean", "Hybrid-Chinese-English", "auto"]

    def is_installed(self) -> bool:
        try:
            import genie_tts
            return True
        except ImportError:
            return False

    def install_dependencies(self):
        logger.info(f"Installing dependencies for {self.id}...")
        os.system("pip install genie-tts soxr")

    def load(self):
        if self._initialized:
            return
        if not self.is_installed():
            logger.error("Genie dependencies not installed.")
            return
        
        if not os.path.exists(self.genie_data_dir):
            os.makedirs(self.genie_data_dir, exist_ok=True)
            
        self._initialized = True

    def _load_character(self, name: str):
        self.load()
        import genie_tts as genie
        from genie_tts.ModelManager import model_manager
        from genie_tts.PredefinedCharacter import CHARA_LANG
        
        if model_manager.has_character(name):
            return True
            
        local_path = None
        for version in ["v2ProPlus", "v2"]:
            path = os.path.join(self.characters_dir, "CharacterModels", version, name)
            if os.path.exists(path):
                local_path = path
                break
        
        if local_path:
            import json
            prompt_wav_json = os.path.join(local_path, "prompt_wav.json")
            if os.path.exists(prompt_wav_json):
                with open(prompt_wav_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                style = "Normal" if "Normal" in data else list(data.keys())[0]
                audio_text = data[style]["text"]
                audio_path = os.path.join(local_path, "prompt_wav", data[style]["wav"])
                lang = CHARA_LANG.get(name, "English")
                
                genie.load_character(
                    character_name=name,
                    onnx_model_dir=os.path.join(local_path, "tts_models"),
                    language=lang
                )
                genie.set_reference_audio(
                    character_name=name,
                    audio_path=audio_path,
                    audio_text=audio_text,
                    language=lang
                )
                return True
        
        # Check if it's a predefined character in genie_tts
        try:
            from genie_tts.PredefinedCharacter import PREDEFINED_CHARACTERS
            if name in PREDEFINED_CHARACTERS:
                genie.load_character(character_name=name)
                return True
        except ImportError:
            pass
            
        return False

    async def generate_stream(self, text: str, voice: str, speed: float, **kwargs) -> AsyncGenerator[np.ndarray, None]:
        import genie_tts as genie
        
        # Check if it's a clone (metadata stored in a sibling file usually, or just assume format)
        is_clone = False
        base_voice = voice
        clone_audio = None
        clone_text = None
        
        # In a real scenario, we'd lookup the clone's metadata
        potential_audio = os.path.join(self.clones_dir, f"{voice}.wav")
        potential_text = os.path.join(self.clones_dir, f"{voice}.txt")
        
        if os.path.exists(potential_audio) and os.path.exists(potential_text):
            is_clone = True
            clone_audio = potential_audio
            with open(potential_text, 'r') as f:
                clone_text = f.read().strip()
            
            # Load metadata if exists
            potential_json = os.path.join(self.clones_dir, f"{voice}.json")
            ref_lang = "English" # Default
            if os.path.exists(potential_json):
                import json
                try:
                    with open(potential_json, 'r') as f:
                        meta = json.load(f)
                        ref_lang = meta.get("ref_lang", "English")
                except: pass
            
            # Find a valid base character instead of hardcoding GPT-SoVITS
            chars = self.get_available_voices()
            local_chars = [c for c in chars if c not in self.list_clones()]
            if "GPT-SoVITS" in chars:
                base_voice = "GPT-SoVITS"
            elif local_chars:
                base_voice = local_chars[0] # Use first available local character (e.g. feibi)
            else:
                base_voice = "GPT-SoVITS" # Fallback
            
        if not self._load_character(base_voice):
            logger.error(f"Genie: Failed to load character {base_voice}")
            return

        if is_clone:
            # We use the reference language from metadata for set_reference_audio
            # to ensure proper prosody extraction, regardless of the target text language.
            genie.set_reference_audio(base_voice, clone_audio, clone_text, ref_lang)
        
        # Prepend language tag if provided
        lang = kwargs.get("lang", "auto")
        tag = ""
        if lang == "English": tag = "[EN]"
        elif lang == "Chinese": tag = "[ZH]"
        elif lang == "Japanese": tag = "[JA]"
        elif lang == "Korean": tag = "[KO]"
        
        if tag and not text.startswith("["):
            text = f"{tag}{text}"
            
        try:
            async for chunk_bytes in genie.tts_async(
                character_name=base_voice,
                text=text,
                play=False,
                split_sentence=True
            ):
                if chunk_bytes:
                    audio_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
                    audio_float32 = audio_int16.astype(np.float32) / 32767.0
                    
                    try:
                        import soxr
                        resampled = soxr.resample(audio_float32, 32000, 24000)
                        yield resampled
                    except ImportError:
                        yield audio_float32
        except Exception as e:
            logger.error(f"Genie streaming error: {e}")

    def generate_batch(self, text: str, voice: str, speed: float, **kwargs) -> Optional[Tuple[int, np.ndarray]]:
        async def _run():
            chunks = []
            async for chunk in self.generate_stream(text, voice, speed, **kwargs):
                if len(chunk) > 0:
                    chunks.append(chunk)
            if chunks:
                return np.concatenate(chunks)
            return None
            
        loop = asyncio.new_event_loop()
        full_audio = loop.run_until_complete(_run())
        loop.close()
        
        if full_audio is not None:
             return (24000, (full_audio * 32767).astype(np.int16))
        return None

    def save_clone(self, name: str, audio_path: str, transcript: Optional[str] = None, **kwargs):
        if not transcript:
            raise ValueError("Genie cloning requires a transcript.")
        
        target_audio = os.path.join(self.clones_dir, f"{name}.wav")
        target_text = os.path.join(self.clones_dir, f"{name}.txt")
        target_json = os.path.join(self.clones_dir, f"{name}.json")
        
        import shutil
        import json
        shutil.copy(audio_path, target_audio)
        with open(target_text, 'w') as f:
            f.write(transcript)
            
        # Save metadata
        ref_lang = kwargs.get("ref_lang", "English")
        with open(target_json, 'w') as f:
            json.dump({"ref_lang": ref_lang}, f)
            
        logger.info(f"Saved Genie clone: {name} with ref_lang: {ref_lang}")

    def list_clones(self) -> List[str]:
        if not os.path.exists(self.clones_dir):
            return []
        return sorted([f.split(".wav")[0] for f in os.listdir(self.clones_dir) if f.endswith(".wav")])
