import json
import os
from kokoro_onnx import Kokoro

class KokoroONNXModel:
    def __init__(self):
        self.model_path = "/home/ubuntu/my-apps/tts-playground/models_data/kokoro-onnx/kokoro-v0_19.onnx"
        self.voices_path = "/home/ubuntu/my-apps/tts-playground/models_data/kokoro-onnx/voices.bin"
        self.voices_info_path = "/home/ubuntu/my-apps/tts-playground/models_data/kokoro-onnx/voices.json"
        self._kokoro = None
        self._voices = None

    @property
    def kokoro(self):
        if self._kokoro is None:
            if not os.path.exists(self.model_path) or not os.path.exists(self.voices_path):
                raise FileNotFoundError(f"ONNX Model or Voices not found at {self.model_path}")
            print(f"Loading Kokoro ONNX from {self.model_path}...")
            self._kokoro = Kokoro(self.model_path, self.voices_path)
        return self._kokoro

    def get_voices(self):
        if self._voices is None:
            if os.path.exists(self.voices_info_path):
                with open(self.voices_info_path, 'r') as f:
                    data = json.load(f)
                    self._voices = sorted(list(data.keys()))
            else:
                self._voices = ["af_heart"]
        return self._voices

kokoro_onnx_model = KokoroONNXModel()
