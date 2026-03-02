import os
from pathlib import Path
from pocket_tts.models.tts_model import TTSModel, _import_model_state
from pocket_tts.default_parameters import DEFAULT_VARIANT
from pocket_tts.modules.stateful_module import init_states

class PocketTTSModel:
    def __init__(self):
        self.model = None
        self.base_dir = "/home/ubuntu/my-apps/tts-playground/models_data/pocket-tts"
        self.local_config = os.path.join(self.base_dir, "b6369a24.yaml")
        self.voice_dir = os.path.join(self.base_dir, "voices")
        self._cached_prompt_states = {}

    def get_model(self):
        if self.model is None:
            config_path = self.local_config if os.path.exists(self.local_config) else DEFAULT_VARIANT
            print(f"🚀 Loading Kyutai Pocket-TTS (from {config_path}) model weights...")
            self.model = TTSModel.load_model(config_path)
            self.model.eval() # CRITICAL for consistent generation
            self.model.to("cpu")
            if not self.model.has_voice_cloning:
                print("⚠️ Warning: Model loaded without voice cloning support!")
        return self.model

    def get_available_voices(self):
        # Local predefined voices in custom folder
        voices = []
        if os.path.exists(self.voice_dir):
            voices += [f.replace(".safetensors", "") for f in os.listdir(self.voice_dir) if f.endswith(".safetensors")]
        return sorted(list(set(voices)))

    def get_prompt_state(self, voice_key: str):
        model = self.get_model()
        if voice_key not in self._cached_prompt_states:
             # Check custom voices folder
             local_voice_path = os.path.join(self.voice_dir, f"{voice_key}.safetensors")
             if os.path.exists(local_voice_path):
                 print(f"📥 Loading local predefined voice: {voice_key}")
                 self._cached_prompt_states[voice_key] = _import_model_state(local_voice_path)
             elif voice_key.endswith(".safetensors") or os.path.exists(voice_key):
                 print(f"📥 Loading custom state: {voice_key}")
                 self._cached_prompt_states[voice_key] = _import_model_state(voice_key)
             else:
                 print(f"📥 Extracting state from audio key: {voice_key}")
                 self._cached_prompt_states[voice_key] = model.get_state_for_audio_prompt(voice_key)
        return self._cached_prompt_states[voice_key]

    def create_clone_state(self, audio_path: str, transcript: str = None):
        """
        Uses the standard library method for voice extraction.
        Pocket-TTS does NOT use transcript for conditioning the voice prompt,
        it uses purely acoustic features from Mimi.
        """
        model = self.get_model()
        print(f"🔊 Extracting voice features from: {os.path.basename(audio_path)}")
        # We use the official library method which is tested and robust
        return model.get_state_for_audio_prompt(audio_path, truncate=True)

pocket_tts_model = PocketTTSModel()
