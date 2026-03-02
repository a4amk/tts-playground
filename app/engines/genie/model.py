import os
import sys
from typing import List, Dict, Optional

# Set up Genie Environment before importing
MODELS_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../models_data/genie"))
GENIE_DATA_DIR = os.path.join(MODELS_BASE, "GenieData")
CHARACTERS_DIR = os.path.join(MODELS_BASE, "characters")

os.environ["GENIE_DATA_DIR"] = GENIE_DATA_DIR

try:
    import genie_tts as genie
    from genie_tts.PredefinedCharacter import CHARA_LANG, CHARA_ALIAS_MAP
    GENIE_AVAILABLE = True
except ImportError:
    GENIE_AVAILABLE = False
    print("Warning: genie-tts not found in venv. Genie engine will be disabled.")

class GenieModel:
    def __init__(self):
        self.initialized = False
        self.base_dir = MODELS_BASE
        self.characters_dir = CHARACTERS_DIR
        self.genie_data_dir = GENIE_DATA_DIR

    def ensure_initialized(self):
        if self.initialized:
            return
        if not GENIE_AVAILABLE:
            raise RuntimeError("genie-tts is not available.")
        
        if not os.path.exists(self.genie_data_dir):
            os.makedirs(self.genie_data_dir, exist_ok=True)
            # We don't auto-download here to avoid long blocking calls in constructor
            # but we assume the setup script or first use will handle it.
            pass
        self.initialized = True

    def get_available_voices(self) -> List[str]:
        """Returns a list of character names available in the local directory."""
        if not os.path.exists(self.characters_dir):
            return []
        
        voices = []
        # Search in v2 and v2ProPlus
        for version in ["v2", "v2ProPlus"]:
            v_dir = os.path.join(self.characters_dir, "CharacterModels", version)
            if os.path.exists(v_dir):
                for chara in os.listdir(v_dir):
                    if os.path.isdir(os.path.join(v_dir, chara)):
                        voices.append(chara)
        
        # Also include predefined aliases if they match what we have
        voices = sorted(list(set(voices)))
        
        # Add user clones from clones_manager
        from ...engines.clones import clones_manager
        clones = clones_manager.list_clones("genie")
        voices.extend(clones)
        return voices

    def load_character(self, name: str):
        self.ensure_initialized()
        
        # Check if already loaded in genie's model_manager
        from genie_tts.ModelManager import model_manager
        if model_manager.has_character(name):
            return True
            
        # Try to find local path
        local_path = None
        version_found = None
        for version in ["v2ProPlus", "v2"]:
            path = os.path.join(self.characters_dir, "CharacterModels", version, name)
            if os.path.exists(path):
                local_path = path
                version_found = version
                break
        
        if local_path:
            # Load from local
            import json
            prompt_wav_json = os.path.join(local_path, "prompt_wav.json")
            if os.path.exists(prompt_wav_json):
                with open(prompt_wav_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Assume 'Normal' or first key
                style = "Normal" if "Normal" in data else list(data.keys())[0]
                audio_text = data[style]["text"]
                audio_path = os.path.join(local_path, "prompt_wav", data[style]["wav"])
                
                # Default language from CHARA_LANG or detect from path/name?
                # For now use English as fallback if not in CHARA_LANG
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
        return False

genie_model = GenieModel()
