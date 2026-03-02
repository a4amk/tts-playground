import os
import torch
import shutil
from typing import List

CLONES_DIR = "/home/ubuntu/my-apps/tts-playground/custom_voices"

class StoredVoicesManager:
    def __init__(self):
        os.makedirs(CLONES_DIR, exist_ok=True)
        # Structure:
        # user_clones/zipvoice/{name}.wav, {name}.txt
        # user_clones/pocket_tts/{name}.pt
        os.makedirs(os.path.join(CLONES_DIR, "zipvoice"), exist_ok=True)
        os.makedirs(os.path.join(CLONES_DIR, "pocket_tts"), exist_ok=True)
        os.makedirs(os.path.join(CLONES_DIR, "genie"), exist_ok=True)
        os.makedirs(os.path.join(CLONES_DIR, "chatterbox"), exist_ok=True)
        os.makedirs(os.path.join(CLONES_DIR, "neutts"), exist_ok=True)

    def save_zipvoice_clone(self, name: str, wav_path: str, text: str):
        dest_wav = os.path.join(CLONES_DIR, "zipvoice", f"{name}.wav")
        dest_txt = os.path.join(CLONES_DIR, "zipvoice", f"{name}.txt")
        shutil.copy(wav_path, dest_wav)
        with open(dest_txt, "w", encoding="utf-8") as f:
            f.write(text)
        return f"ZipVoice clone '{name}' saved."

    def save_genie_clone(self, name: str, wav_path: str, text: str, base_voice: str):
        dest_wav = os.path.join(CLONES_DIR, "genie", f"{name}.wav")
        dest_txt = os.path.join(CLONES_DIR, "genie", f"{name}.txt")
        dest_base = os.path.join(CLONES_DIR, "genie", f"{name}.base")
        shutil.copy(wav_path, dest_wav)
        with open(dest_txt, "w", encoding="utf-8") as f:
            f.write(text)
        with open(dest_base, "w", encoding="utf-8") as f:
            f.write(base_voice)
        return f"Genie clone '{name}' (based on '{base_voice}') saved."

    def save_pocket_tts_clone(self, name: str, model_state, export_fn):
        # export_fn is pocket_tts.export_model_state
        dest_sf = os.path.join(CLONES_DIR, "pocket_tts", f"{name}.safetensors")
        export_fn(model_state, dest_sf)
        return f"Pocket-TTS clone '{name}' saved as safetensors."

    def save_chatterbox_clone(self, name: str, wav_path: str):
        dest_wav = os.path.join(CLONES_DIR, "chatterbox", f"{name}.wav")
        shutil.copy(wav_path, dest_wav)
        return f"Chatterbox clone '{name}' saved."

    def get_zipvoice_clone_data(self, name: str):
        # Returns (wav_path, transcript)
        # name might have .wav already if coming from list_clones
        base = name.replace(".wav", "")
        wav_path = os.path.join(CLONES_DIR, "zipvoice", f"{base}.wav")
        txt_path = os.path.join(CLONES_DIR, "zipvoice", f"{base}.txt")
        transcript = ""
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                transcript = f.read().strip()
        return wav_path, transcript

    def list_clones(self, engine_id: str) -> List[str]:
        engine_map = {
            "zipvoice": "zipvoice",
            "pocket-tts": "pocket_tts",
            "genie": "genie",
            "chatterbox-turbo-onnx": "chatterbox",
            "neutts": "neutts"
        }
        folder = engine_map.get(engine_id)
        if not folder:
            return []
        
        path = os.path.join(CLONES_DIR, folder)
        clones = []
        if not os.path.exists(path): return []
        for f in os.listdir(path):
            if f.endswith(".wav") and engine_id in ["zipvoice", "genie", "chatterbox-turbo-onnx", "neutts"]:
                clones.append(f)
            elif (f.endswith(".safetensors") or f.endswith(".pt")) and engine_id == "pocket-tts":
                name = f.replace(".safetensors", "").replace(".pt", "")
                if name not in clones:
                    clones.append(name)
        return clones

    def get_genie_clone_data(self, name: str):
        base = name.replace(".wav", "")
        wav_path = os.path.join(CLONES_DIR, "genie", f"{base}.wav")
        txt_path = os.path.join(CLONES_DIR, "genie", f"{base}.txt")
        base_path = os.path.join(CLONES_DIR, "genie", f"{base}.base")
        transcript = ""
        base_voice = "thirtyseven"
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                transcript = f.read().strip()
        if os.path.exists(base_path):
            with open(base_path, "r", encoding="utf-8") as f:
                base_voice = f.read().strip()
        return wav_path, transcript, base_voice

    def get_clone_path(self, engine_id: str, name: str):
        if engine_id in ["zipvoice", "genie", "chatterbox-turbo-onnx", "neutts"]:
            base = name.replace(".wav", "")
            if engine_id == "zipvoice":
                path = os.path.join(CLONES_DIR, "zipvoice", f"{base}.wav")
            elif engine_id == "genie":
                path = os.path.join(CLONES_DIR, "genie", f"{base}.wav")
            elif engine_id == "neutts":
                path = os.path.join(CLONES_DIR, "neutts", f"{base}.wav")
            else:
                path = os.path.join(CLONES_DIR, "chatterbox", f"{base}.wav")
            if os.path.exists(path):
                 return path
        elif engine_id == "pocket-tts":
            # Prefer safetensors
            for ext in [".safetensors", ".pt"]:
                path = os.path.join(CLONES_DIR, "pocket_tts", f"{name}{ext}")
                if os.path.exists(path):
                    return path
        return None

clones_manager = StoredVoicesManager()
