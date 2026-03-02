import os
from kokoro import KPipeline

class KokoroModel:
    """
    Handles Kokoro weights/checkpoints loading and asset discovery (voices).
    This is the 'Model' part of the separation.
    """
    def __init__(self):
        os.environ["ESPEAK_DATA_PATH"] = "/usr/lib/aarch64-linux-gnu/espeak-ng-data"
        self.pipelines = {}
        self.voice_dir = "/home/ubuntu/my-apps/tts-playground/models_data/kokoro-82m/voices"

    def get_pipeline(self, lang_code: str):
        if lang_code not in self.pipelines:
            print(f"Loading KPipeline for lang_code '{lang_code}'...")
            try:
                self.pipelines[lang_code] = KPipeline(lang_code=lang_code)
            except Exception as e:
                if lang_code != 'a':
                    return self.get_pipeline('a')
                raise e
        return self.pipelines[lang_code]

    def get_voices(self):
        available_voices = ["af_heart"]
        if os.path.exists(self.voice_dir):
            voices = sorted([f.split('.')[0] for f in os.listdir(self.voice_dir) if f.endswith('.pt')])
            if voices:
                available_voices = voices
        return available_voices

# Global singleton or per-instance
kokoro_model = KokoroModel()
