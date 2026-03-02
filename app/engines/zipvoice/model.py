import os
import torch
import sys

# Hack to import ZipVoice from external model directory
zipvoice_dir = "/home/ubuntu/my-apps/tts-playground/models_data/zipvoice/ZipVoice"
if zipvoice_dir not in sys.path:
    sys.path.append(zipvoice_dir)

from zipvoice.models.zipvoice import ZipVoice
from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
from vocos import Vocos
from huggingface_hub import hf_hub_download
from zipvoice.utils.feature import VocosFbank
import json

class ZipVoiceModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # We enforce CPU for consistency in this playground unless otherwise needed, but ZipVoice is fast.
        self.device = torch.device("cpu")
        self.model_name = "zipvoice_distill" # Faster one
        self.tokenizer = None
        self.feature_extractor = None
        self.model = None
        self.vocoder = None
        self.base_dir = "/home/ubuntu/my-apps/tts-playground/models_data/zipvoice"
        self.ref_dir = os.path.join(self.base_dir, "references")
        os.makedirs(self.ref_dir, exist_ok=True)

    def load_model(self):
        if self.model is not None:
            return

        print("🚀 Loading ZipVoice Engine...")
        
        # Download files if needed
        model_ckpt = hf_hub_download("k2-fsa/ZipVoice", filename=f"{self.model_name}/model.pt")
        model_config_path = hf_hub_download("k2-fsa/ZipVoice", filename=f"{self.model_name}/model.json")
        token_file = hf_hub_download("k2-fsa/ZipVoice", filename=f"{self.model_name}/tokens.txt")
        
        self.tokenizer = EmiliaTokenizer(token_file=token_file)
        
        with open(model_config_path, "r") as f:
            model_config = json.load(f)
            
        tokenizer_config = {"vocab_size": self.tokenizer.vocab_size, "pad_id": self.tokenizer.pad_id}
        
        if self.model_name == "zipvoice":
            self.model = ZipVoice(**model_config["model"], **tokenizer_config)
        else:
            self.model = ZipVoiceDistill(**model_config["model"], **tokenizer_config)
            
        state_dict = torch.load(model_ckpt, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict["model"], strict=True)
        self.model.eval().to(self.device)
        
        self.vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(self.device).eval()
        self.feature_extractor = VocosFbank()

    def get_available_voices(self):
        if not os.path.exists(self.ref_dir):
            return ["default_ref.wav"]
        voices = [f for f in os.listdir(self.ref_dir) if f.endswith((".wav", ".mp3", ".flac"))]
        if not voices:
            # create a dummy
            import numpy as np, soundfile as sf
            t = np.linspace(0, 3, 3*24000)
            sf.write(os.path.join(self.ref_dir, "default_ref.wav"), np.sin(2*np.pi*440*t), 24000)
            with open(os.path.join(self.ref_dir, "default_ref.txt"), "w") as f:
                f.write("Beep.")
            voices = ["default_ref.wav"]
        return sorted(voices)
        
zipvoice_model = ZipVoiceModel()
