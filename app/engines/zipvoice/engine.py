import os
import json
import torch
import numpy as np
import logging
import time
import asyncio
import librosa
import sys
import datetime as dt
from typing import AsyncGenerator, Tuple, Optional, List, Dict, Any

from ..interface import TTSPlugin
from ...config import get_device, HF_HUB_OFFLINE, ZIPVOICE_USE_ONNX
from ...utils import secure_path_join
import onnxruntime as ort

logger = logging.getLogger(__name__)

# External ZipVoice paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
zipvoice_dir = os.path.join(project_root, "models_data/zipvoice/ZipVoice")
if zipvoice_dir not in sys.path:
    sys.path.append(zipvoice_dir)

try:
    from zipvoice.models.zipvoice import ZipVoice
    from zipvoice.models.zipvoice_distill import ZipVoiceDistill
    from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
    from vocos import Vocos
    from huggingface_hub import hf_hub_download
    from zipvoice.utils.feature import VocosFbank
    from zipvoice.utils.infer import (
        add_punctuation,
        chunk_tokens_punctuation,
        load_prompt_wav,
        remove_silence,
        rms_norm,
    )
    import json
except ImportError:
    pass

class ZipVoiceEngine(TTSPlugin):
    """
    Standardized ZipVoice Engine.
    """
    def __init__(self):
        self._id = "zipvoice"
        self._display_name = "ZipVoice (Distilled Flow Matching)"
        
        self.device = get_device("cpu")
        self.model_name = "zipvoice_distill"
        
        self.tokenizer = None
        self.feature_extractor = None
        self.model = None
        self.vocoder = None
        self.use_onnx = ZIPVOICE_USE_ONNX
        self.onnx_text_encoder = None
        self.onnx_fm_decoder = None
        self._current_quant = None
        
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        self.base_dir = os.path.join(project_root, "models_data/zipvoice")
        self.ref_dir = os.path.join(self.base_dir, "references")
        self.clones_dir = os.path.join(self.base_dir, "custom_voices", self._id)
        
        os.makedirs(self.ref_dir, exist_ok=True)
        os.makedirs(self.clones_dir, exist_ok=True)
        
        self.prompt_cache = {}
        self.lock = threading.Lock() # Override base lock if needed, but TTSPlugin has it now

    def get_extra_controls(self) -> List[Dict[str, Any]]:
        return []

    def get_standard_controls(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "speed",
                "label": "Playback Speed",
                "info": "Adjusts the relative tempo of the generated speech. Higher = faster, Lower = slower. Works with both streaming and batch generation.",
                "min": 0.5, "max": 2.0, "step": 0.1, "default": 1.0
            }
        ]

    def get_variants(self) -> List[Dict[str, Any]]:
        return [
            {"id": "pytorch", "label": "PyTorch (FP32)", "default": not ZIPVOICE_USE_ONNX},
            {"id": "onnx_fp32", "label": "ONNX (FP32)", "default": ZIPVOICE_USE_ONNX},
            {"id": "onnx_int8", "label": "ONNX (INT8)", "default": False}
        ]

    @property
    def id(self) -> str:
        return self._id

    @property
    def display_name(self) -> str:
        return self._display_name

    def get_ui_config(self) -> Dict[str, Any]:
        return {
            "temp": 0.7,
            "speed": 1.0
        }

    def get_cloning_config(self) -> Dict[str, Any]:
        return {
            "requires_cloning": True,
            "requires_transcript": True,
            "instruction": "Upload Reference Audio AND exact Reference Text (Transcript)."
        }

    def get_available_voices(self) -> List[str]:
        voices = []
        if os.path.exists(self.ref_dir):
            voices += [f for f in os.listdir(self.ref_dir) if f.endswith((".wav", ".mp3", ".flac"))]
            
        clones = self.list_clones()
        combined = voices + clones
        
        if not combined:
            return ["default_ref.wav"]
            
        return sorted(list(set(combined)))

    def get_available_languages(self) -> List[str]:
        return ["auto"]

    def is_installed(self) -> bool:
        try:
            import vocos
            return True
        except ImportError:
            return False

    def install_dependencies(self):
        logger.info(f"Installing dependencies for {self.id}...")
        os.system("pip install vocos soundfile")

    def load(self, variant: Optional[str] = None):
        # Determine settings from variant
        if variant == "pytorch":
            use_onnx = False
            quant = "FP32"
        elif variant == "onnx_int8":
            use_onnx = True
            quant = "INT8"
        elif variant == "onnx_fp32":
            use_onnx = True
            quant = "FP32"
        else:
            # Fallback to env default
            use_onnx = ZIPVOICE_USE_ONNX
            quant = self._current_quant or "FP32"

        # If already loaded with different settings, we should ideally reload
        if (self.model is not None or self.onnx_text_encoder is not None):
             if (self.use_onnx != use_onnx) or (use_onnx and self._current_quant != quant):
                 logger.info(f"Reloading ZipVoice with variant={variant} (ONNX={use_onnx}, Quant={quant})...")
                 self.onnx_text_encoder = None
                 self.onnx_fm_decoder = None
                 self.model = None
                 self.use_onnx = use_onnx
             else:
                 return

        self.use_onnx = use_onnx
        logger.info(f"loading ZipVoice Engine (Variant={variant}, ONNX={self.use_onnx})...")
        kwargs = {"local_files_only": HF_HUB_OFFLINE}
        
        # Common files
        token_file = hf_hub_download("k2-fsa/ZipVoice", filename=f"{self.model_name}/tokens.txt", **kwargs)
        self.tokenizer = EmiliaTokenizer(token_file=token_file)
        
        if self.use_onnx:
            # Load ONNX models
            suffix = "" if quant == "FP32" else "_int8"
            
            text_encoder_path = hf_hub_download("k2-fsa/ZipVoice", filename=f"{self.model_name}/text_encoder{suffix}.onnx", **kwargs)
            fm_decoder_path = hf_hub_download("k2-fsa/ZipVoice", filename=f"{self.model_name}/fm_decoder{suffix}.onnx", **kwargs)
            
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 4
            self.onnx_text_encoder = ort.InferenceSession(text_encoder_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])
            self.onnx_fm_decoder = ort.InferenceSession(fm_decoder_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])
            self._current_quant = quant
            
            # Get feat_dim from metadata
            meta = self.onnx_fm_decoder.get_modelmeta().custom_metadata_map
            self.feat_dim = int(meta.get("feat_dim", 100))
        else:
            # Load PyTorch models
            model_ckpt = hf_hub_download("k2-fsa/ZipVoice", filename=f"{self.model_name}/model.pt", **kwargs)
            model_config_path = hf_hub_download("k2-fsa/ZipVoice", filename=f"{self.model_name}/model.json", **kwargs)
            
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
            self.feat_dim = model_config["model"]["feat_dim"]
        
        self.vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(self.device).eval()
        self.feature_extractor = VocosFbank()

    def _get_ref_path(self, voice: str) -> str:
        # Check custom
        try:
            potential_path = secure_path_join(self.clones_dir, voice if voice.endswith(".wav") else f"{voice}.wav")
            if os.path.exists(potential_path):
                return potential_path
        except ValueError:
            pass # Blocked traversal
            
        # Check internal ref dir
        try:
            return secure_path_join(self.ref_dir, voice)
        except ValueError:
             # If both fail or are invalid, fallback to a default if available
             return os.path.join(self.ref_dir, "default.wav")

    async def generate_stream(self, text: str, voice: str, speed: float, variant: Optional[str] = None, **kwargs) -> AsyncGenerator[np.ndarray, None]:
        self.load(variant=variant)
        
        ref_path = self._get_ref_path(voice)
        txt_path = os.path.splitext(ref_path)[0] + ".txt"
        
        prompt_text = ""
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()
                
        def _generate():
            sampling_rate = 24000
            feat_scale = 0.1
            target_rms = 0.1
            
            cache_key = (voice, prompt_text)
            with self.lock:
                if cache_key in self.prompt_cache:
                    prompt_features, prompt_rms = self.prompt_cache[cache_key]
                else:
                    prompt_wav = load_prompt_wav(ref_path, sampling_rate=sampling_rate)
                    prompt_wav = remove_silence(prompt_wav, sampling_rate, only_edge=False, trail_sil=200)
                    prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)
                    
                    prompt_features = self.feature_extractor.extract(prompt_wav, sampling_rate=sampling_rate).to(self.device)
                    prompt_features = prompt_features.unsqueeze(0) * feat_scale
                    self.prompt_cache[cache_key] = (prompt_features, prompt_rms)
            
            pt = add_punctuation(prompt_text)
            prompt_tokens_str = self.tokenizer.texts_to_tokens([pt])[0]
            prompt_tokens = self.tokenizer.tokens_to_token_ids([prompt_tokens_str])

            split_choice = kwargs.get("split_choice", "Sentences (Punctuation)")
            custom_regex = kwargs.get("custom_regex", r'\n+')
            chunk_texts = self.split_text(text, split_choice, custom_regex)
            
            for chunk_text in chunk_texts:
                t = add_punctuation(chunk_text)
                tokens_str = self.tokenizer.texts_to_tokens([t])[0]
                chunked_token_sequences_str = chunk_tokens_punctuation(tokens_str, max_tokens=50)
                
                for sub_tokens_str in chunked_token_sequences_str:
                    sub_token_ids = self.tokenizer.tokens_to_token_ids([sub_tokens_str])[0]
                    
                    with torch.inference_mode():
                        self.load(variant=variant)
                        
                        if self.use_onnx:
                            # ONNX path
                            onnx_tokens = torch.tensor([sub_token_ids], dtype=torch.int64)
                            onnx_prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.int64)
                            onnx_prompt_features_len = torch.tensor(prompt_features.size(1), dtype=torch.int64)
                            onnx_speed = torch.tensor(speed, dtype=torch.float32)

                            text_condition = self.onnx_text_encoder.run(None, {
                                self.onnx_text_encoder.get_inputs()[0].name: onnx_tokens.numpy(),
                                self.onnx_text_encoder.get_inputs()[1].name: onnx_prompt_tokens.numpy(),
                                self.onnx_text_encoder.get_inputs()[2].name: onnx_prompt_features_len.numpy(),
                                self.onnx_text_encoder.get_inputs()[3].name: onnx_speed.numpy()
                            })[0]
                            text_condition = torch.from_numpy(text_condition)

                            # FM Decoder sample loop
                            num_step = 8 if self.model_name == "zipvoice_distill" else 16
                            t_shift = 0.5
                            guidance_val = 3.0 if self.model_name == "zipvoice_distill" else 1.0
                            
                            from zipvoice.models.modules.solver import get_time_steps
                            timesteps = get_time_steps(t_start=0.0, t_end=1.0, num_step=num_step, t_shift=t_shift)
                            
                            batch_size, num_frames, _ = text_condition.shape
                            x = torch.randn(batch_size, num_frames, self.feat_dim)
                            speech_condition = torch.nn.functional.pad(prompt_features, (0, 0, 0, num_frames - prompt_features.shape[1]))
                            guidance_scale = torch.tensor(guidance_val, dtype=torch.float32)

                            for stp in range(num_step):
                                v = self.onnx_fm_decoder.run(None, {
                                    self.onnx_fm_decoder.get_inputs()[0].name: timesteps[stp].numpy(),
                                    self.onnx_fm_decoder.get_inputs()[1].name: x.numpy(),
                                    self.onnx_fm_decoder.get_inputs()[2].name: text_condition.numpy(),
                                    self.onnx_fm_decoder.get_inputs()[3].name: speech_condition.numpy(),
                                    self.onnx_fm_decoder.get_inputs()[4].name: guidance_scale.numpy()
                                })[0]
                                x = x + torch.from_numpy(v) * (timesteps[stp + 1] - timesteps[stp])

                            pred_features = x[:, prompt_features.size(1):, :]
                            pred_features_lens = torch.tensor([pred_features.size(1)], dtype=torch.int64)
                        else:
                            # PyTorch path
                            batch_prompt_features_lens = torch.full((1,), prompt_features.size(1), device=self.device)
                            (
                                pred_features,
                                pred_features_lens,
                                _, _
                            ) = self.model.sample(
                                tokens=[sub_token_ids],
                                prompt_tokens=prompt_tokens,
                                prompt_features=prompt_features,
                                prompt_features_lens=batch_prompt_features_lens,
                                speed=speed,
                                t_shift=0.5,
                                duration="predict",
                                num_step=8 if self.model_name == "zipvoice_distill" else 16,
                                guidance_scale=3.0 if self.model_name == "zipvoice_distill" else 1.0,
                            )
                        
                        pred_features = pred_features.permute(0, 2, 1) / feat_scale
                        wav = self.vocoder.decode(pred_features[0][None, :, : pred_features_lens[0]]).squeeze(1).clamp(-1, 1)
                        
                        if prompt_rms < target_rms:
                            wav = wav * prompt_rms / target_rms
                        
                        wav_np = wav.cpu().numpy().flatten().astype(np.float32)
                        if speed != 1.0:
                             wav_np = librosa.effects.time_stretch(wav_np, rate=speed)
                             
                        yield wav_np

        import queue
        import threading
        q = queue.Queue()
        
        def worker():
            try:
                for chunk in _generate():
                    q.put(("data", chunk))
            except Exception as e:
                q.put(("error", str(e)))
            finally:
                q.put(("done", None))

        thread = threading.Thread(target=worker)
        thread.start()
        
        while True:
            msg_type, chunk = await asyncio.get_event_loop().run_in_executor(None, q.get)
            if msg_type == "done": break
            if msg_type == "error":
                logger.error(f"ZipVoice stream error: {chunk}")
                break
            if msg_type == "data" and chunk is not None:
                yield chunk

    def generate_batch(self, text: str, voice: str, speed: float, variant: Optional[str] = None, **kwargs) -> Optional[Tuple[int, np.ndarray]]:
        async def _run():
            chunks = []
            async for chunk in self.generate_stream(text, voice, speed, variant=variant, **kwargs):
                if len(chunk) > 0: chunks.append(chunk)
            if chunks:
                return np.concatenate(chunks)
            return None
            
        loop = asyncio.new_event_loop()
        full_audio = loop.run_until_complete(_run())
        loop.close()
        
        if full_audio is not None:
            return 24000, (full_audio * 32767).astype(np.int16)
        return None

    def save_clone(self, name: str, audio_path: str, transcript: Optional[str] = None, **kwargs):
        if not transcript:
            raise ValueError("ZipVoice cloning requires a transcript.")
            
        target_audio = os.path.join(self.clones_dir, f"{name}.wav")
        target_txt = os.path.join(self.clones_dir, f"{name}.txt")
        
        target_json = os.path.join(self.clones_dir, f"{name}.json")
        
        import shutil
        import json
        shutil.copy(audio_path, target_audio)
        with open(target_txt, "w", encoding="utf-8") as f:
            f.write(transcript)
            
        # Save metadata
        ref_lang = kwargs.get("ref_lang", "English")
        with open(target_json, 'w') as f:
            json.dump({"ref_lang": ref_lang}, f)
            
        logger.info(f"Saved ZipVoice clone: {name} with ref_lang: {ref_lang}")

    def list_clones(self) -> List[str]:
        if not os.path.exists(self.clones_dir):
            return []
        return sorted([f for f in os.listdir(self.clones_dir) if f.endswith(".wav")])
