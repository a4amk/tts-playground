import os
from piper.voice import PiperVoice

class PiperModel:
    def __init__(self):
        self.voices_base = "/home/ubuntu/my-apps/tts-playground/models_data/piper-onnx"
        self._loaded_voices = {}

    def get_voice_pipeline(self, voice_key: str):
        if voice_key not in self._loaded_voices:
            # Look for the onnx file and its json in models_data/piper/{voice_key}/
            voice_dir = os.path.join(self.voices_base, voice_key)
            if not os.path.exists(voice_dir):
                 return None
                 
            onnx_files = [f for f in os.listdir(voice_dir) if f.endswith(".onnx")]
            if not onnx_files:
                 return None
            
            model_path = os.path.join(voice_dir, onnx_files[0])
            config_path = model_path + ".json"
            
            if not os.path.exists(config_path):
                 # Try finding any .json in the folder
                 json_files = [f for f in os.listdir(voice_dir) if f.endswith(".json")]
                 if json_files:
                      config_path = os.path.join(voice_dir, json_files[0])
            
            print(f"Loading Piper voice from {model_path} (config: {config_path})...")
            # Piper lib can load by model path if json is same-name.onnx.json
            self._loaded_voices[voice_key] = PiperVoice.load(model_path, config_path=config_path)
            
        return self._loaded_voices[voice_key]

    def get_available_voices(self):
        if not os.path.exists(self.voices_base):
             return []
        # Return voices that have an .onnx file
        voices = []
        for d in os.listdir(self.voices_base):
             voice_dir = os.path.join(self.voices_base, d)
             if os.path.isdir(voice_dir):
                  if any(f.endswith(".onnx") for f in os.listdir(voice_dir)):
                       voices.append(d)
        return sorted(voices)

piper_model = PiperModel()
