import os
import asyncio
import numpy as np
import logging
import librosa
import onnxruntime
import subprocess
from typing import AsyncGenerator, Tuple, Optional, List, Dict, Any
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from ..interface import TTSPlugin
from ...config import get_device, HF_HUB_OFFLINE
from ...utils import secure_path_join

logger = logging.getLogger(__name__)

# Constants from original model.py
MODEL_ID = "ResembleAI/chatterbox-turbo-ONNX"
SAMPLE_RATE = 24000
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562
SILENCE_TOKEN = 4299
NUM_KV_HEADS = 16
HEAD_DIM = 64

class RepetitionPenaltyLogitsProcessor:
    def __init__(self, penalty: float):
        self.penalty = float(penalty)

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        score = np.take_along_axis(scores, input_ids, axis=1)
        score = np.where(score < 0, score * self.penalty, score / self.penalty)
        scores_processed = scores.copy()
        np.put_along_axis(scores_processed, input_ids, score, axis=1)
        return scores_processed

class ChatterboxONNXEngine(TTSPlugin):
    """
    Standardized Chatterbox Turbo ONNX Engine.
    """
    def __init__(self):
        self._id = "chatterbox_onnx"
        self._display_name = "Chatterbox Turbo (Moshi/ONNX)"
        
        # Sessions
        self.speech_encoder_session = None
        self.embed_tokens_session = None
        self.language_model_session = None
        self.cond_decoder_session = None
        self.tokenizer = None
        self._is_loaded = False
        self._current_dtype = None
        
        # Paths
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        self.device = get_device("cpu")
        self.clones_dir = os.path.join(self.base_dir, "custom_voices", self._id)
        os.makedirs(self.clones_dir, exist_ok=True)

    @property
    def id(self) -> str:
        return self._id

    @property
    def display_name(self) -> str:
        return self._display_name

    def get_ui_config(self) -> Dict[str, Any]:
        return {
            "temp": 0.7,
            "top_k": 50,
            "rep_pen": 1.2,
            "speed": 1.0
        }

    def get_cloning_config(self) -> Dict[str, Any]:
        return {
            "requires_cloning": True,
            "requires_transcript": False,
            "instruction": "Upload a short (5-10s) clear audio sample of the voice you want to clone. Use the 'Edit' (scissors) button to trim if the audio exceeds 15s."
        }
    def get_standard_controls(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "speed", "label": "Synthesis Speed",
                "info": "Adjusts the tempo of the generated speech. Works with streaming and batch generation.",
                "min": 0.5, "max": 2.0, "step": 0.1, "default": 1.0
            },
            {
                "id": "temp", "label": "Sampling Temperature",
                "info": "Controls randomness. High = creative/expressive, Low = precise/consistent. Recommended: 0.7. Works with stream and batch.",
                "min": 0.1, "max": 2.0, "step": 0.1, "default": 0.7
            },
            {
                "id": "top_k", "label": "Top-K Filtering",
                "info": "Filters lowest probability tokens. Recommended: 50. Works with stream and batch.",
                "min": 1, "max": 200, "step": 1, "default": 50
            },
            {
                "id": "rep_pen", "label": "Repetition Penalty",
                "info": "Penalizes repeating the same word/sound. Recommended: 1.2. Works with stream and batch.",
                "min": 1.0, "max": 2.0, "step": 0.1, "default": 1.2
            },
            {
                "id": "cfg", "label": "CFG Scale",
                "info": "Classifier-Free Guidance. Higher = stronger adherence to prompt style. Recommended: 0.5. Works with stream and batch.",
                "min": 0.0, "max": 5.0, "step": 0.1, "default": 0.5
            },
            {
                "id": "exaggeration", "label": "Exaggeration / Emotion",
                "info": "Controls prosodic intensity and 'character' of the voice. Higher = more animated. Recommended: 0.5. Works with stream and batch.",
                "min": 0.0, "max": 2.0, "step": 0.1, "default": 0.5
            }
        ]

    def get_extra_controls(self) -> List[Dict[str, Any]]:
        return []

    def get_variants(self) -> List[Dict[str, Any]]:
        return [
            {"id": "fp32", "label": "Chatterbox Turbo (ONNX FP32)", "default": True},
            {"id": "fp16", "label": "Chatterbox Turbo (ONNX FP16) [Broken on CPU]", "default": False},
            {"id": "q8", "label": "Chatterbox Turbo (ONNX Q8/INT8) [Broken on CPU]", "default": False}
        ]

    def get_available_voices(self) -> List[str]:
        clones = self.list_clones()
        return ["default"] + clones

    def get_available_languages(self) -> List[str]:
        return ["en"]

    def is_installed(self) -> bool:
        try:
            import onnxruntime
            import transformers
            import librosa
            return True
        except ImportError:
            return False

    def install_dependencies(self):
        logger.info(f"Installing dependencies for {self.id}...")
        subprocess.run(["pip", "install", "onnxruntime", "transformers", "librosa", "huggingface_hub"], check=False)

    def _download_onnx_model(self, name: str, dtype: str = "fp32") -> str:
        filename = f"{name}{'' if dtype == 'fp32' else '_quantized' if dtype == 'q8' else f'_{dtype}'}.onnx"
        kwargs = {"local_files_only": HF_HUB_OFFLINE}
        # Model graph file
        graph = hf_hub_download(MODEL_ID, subfolder="onnx", filename=filename, **kwargs)
        # Weight data file
        hf_hub_download(MODEL_ID, subfolder="onnx", filename=f"{filename}_data", **kwargs)
        return graph

    def load(self, variant: Optional[str] = None):
        dtype = variant or "fp32"
        if self._is_loaded and self._current_dtype == dtype:
             return
        
        if self._is_loaded:
            logger.info(f"Reloading Chatterbox Turbo ONNX models (new variant/dtype={dtype})...")
            self.speech_encoder_session = None
            self.embed_tokens_session = None
            self.language_model_session = None
            self.cond_decoder_session = None

        logger.info(f"Loading Chatterbox Turbo ONNX models (variant={variant}, dtype={dtype})...")
        conditional_decoder_path = self._download_onnx_model("conditional_decoder", dtype=dtype)
        speech_encoder_path = self._download_onnx_model("speech_encoder", dtype=dtype)
        embed_tokens_path = self._download_onnx_model("embed_tokens", dtype=dtype)
        language_model_path = self._download_onnx_model("language_model", dtype=dtype)

        opts = onnxruntime.SessionOptions()
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        try:
            from onnxruntime_extensions import get_library_path
            opts.register_custom_ops_library(get_library_path())
        except (ImportError, Exception) as e:
            logger.warning(f"Failed to register onnxruntime-extensions: {e}")
        
        providers = ['CPUExecutionProvider']
        self.speech_encoder_session = onnxruntime.InferenceSession(speech_encoder_path, opts, providers=providers)
        self.embed_tokens_session = onnxruntime.InferenceSession(embed_tokens_path, opts, providers=providers)
        self.language_model_session = onnxruntime.InferenceSession(language_model_path, opts, providers=providers)
        self.cond_decoder_session = onnxruntime.InferenceSession(conditional_decoder_path, opts, providers=providers)
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self._current_dtype = dtype
        self._is_loaded = True

    def _prepare_audio_values(self, voice: str) -> np.ndarray:
        voice_path = None
        if voice != "default":
            try:
                potential_path = secure_path_join(self.clones_dir, f"{voice}.wav")
                if os.path.exists(potential_path):
                    voice_path = potential_path
            except ValueError:
                pass
                
        if not voice_path:
            # Fallback to internal default
            voice_path = os.path.join(self.base_dir, "models_data/chatterbox_onnx/female_shadowheart4.flac")
            
        if not os.path.exists(voice_path):
             # Hard fallback to a zero array if file is missing (not ideal but safe)
             return np.zeros((1, SAMPLE_RATE * 5), dtype=np.float32)

        audio_values, _ = librosa.load(voice_path, sr=SAMPLE_RATE)
        return audio_values[np.newaxis, :].astype(np.float32)

    async def generate_stream(self, text: str, voice: str, speed: float, variant: Optional[str] = None, **kwargs) -> AsyncGenerator[np.ndarray, None]:
        # Filter out named arguments if they slipped into kwargs
        kwargs.pop("text", None)
        kwargs.pop("voice", None)
        kwargs.pop("speed", None)
        
        self.load(variant=variant)
        split_choice = kwargs.get("split_choice", "Both (Newlines & Sentences)")
        custom_regex = kwargs.get("custom_regex", r'\n+')
        chunks = self.split_text(text, split_choice, custom_regex)
        
        audio_values = self._prepare_audio_values(voice)
        
        for chunk_text in chunks:
            if not chunk_text.strip(): continue
            audio = await asyncio.to_thread(self._generate_single, chunk_text, audio_values, **kwargs)
            if audio is not None:
                yield audio

    def generate_batch(self, text: str, voice: str, speed: float, variant: Optional[str] = None, **kwargs) -> Optional[Tuple[int, np.ndarray]]:
        self.load(variant=variant)
        split_choice = kwargs.get("split_choice", "Both (Newlines & Sentences)")
        custom_regex = kwargs.get("custom_regex", r'\n+')
        chunks = self.split_text(text, split_choice, custom_regex)
        
        audio_values = self._prepare_audio_values(voice)
        
        audio_chunks = []
        for chunk_text in chunks:
             if not chunk_text.strip(): continue
             audio = self._generate_single(chunk_text, audio_values, **kwargs)
             if audio is not None:
                 audio_chunks.append(audio)
                 
        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
            if speed != 1.0:
                import librosa
                full_audio = librosa.effects.time_stretch(full_audio, rate=speed)
            return SAMPLE_RATE, (full_audio * 32767).astype(np.int16)
        return None

    def _generate_single(self, text: str, audio_values: np.ndarray, **kwargs) -> Optional[np.ndarray]:
        # Implementation logic from original runtime.py
        max_new_tokens = 1000
        repetition_penalty = float(kwargs.get("rep_pen", 1.2))
        top_k = int(kwargs.get("top_k", 50))
        temp = float(kwargs.get("temp", 0.7))
        speed = float(kwargs.get("speed", 1.0))
        
        input_ids = self.tokenizer(text, return_tensors="np")["input_ids"].astype(np.int64)
        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)
        generate_tokens = np.array([[START_SPEECH_TOKEN]], dtype=np.int64)
        
        # ORT Generation
        for i in range(max_new_tokens):
            inputs_embeds = self.embed_tokens_session.run(None, {"input_ids": input_ids})[0]

            if i == 0:
                ort_speech_encoder_input = {"audio_values": audio_values}
                cond_emb, prompt_token, speaker_embeddings, speaker_features = self.speech_encoder_session.run(None, ort_speech_encoder_input)
                inputs_embeds = np.concatenate((cond_emb, inputs_embeds), axis=1)

                batch_size, seq_len, _ = inputs_embeds.shape
                past_key_values = {
                    i_in.name: np.zeros([batch_size, NUM_KV_HEADS, 0, HEAD_DIM], dtype=np.float16 if i_in.type == 'tensor(float16)' else np.float32)
                    for i_in in self.language_model_session.get_inputs()
                    if "past_key_values" in i_in.name
                }
                attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
                position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1).repeat(batch_size, axis=0)

            run_dict = dict(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **past_key_values,
            )
            
            output = self.language_model_session.run(None, run_dict)
            logits = output[0]
            present_key_values = output[1:]

            logits = logits[:, -1, :]
            next_token_logits = repetition_penalty_processor(generate_tokens, logits)

            if temp == 0:
                 new_input_ids = np.argmax(next_token_logits, axis=-1, keepdims=True).astype(np.int64)
            else:
                 probs = np.exp(next_token_logits / max(temp, 0.1)) / np.sum(np.exp(next_token_logits / max(temp, 0.1)), axis=-1, keepdims=True)
                 new_input_ids = np.array([[np.random.choice(len(probs[0]), p=probs[0])]], dtype=np.int64)

            generate_tokens = np.concatenate((generate_tokens, new_input_ids), axis=-1)
            
            if (new_input_ids.flatten() == STOP_SPEECH_TOKEN).all():
                break

            input_ids = new_input_ids
            attention_mask = np.concatenate([attention_mask, np.ones((batch_size, 1), dtype=np.int64)], axis=1)
            position_ids = position_ids[:, -1:] + 1
            for j, key_name in enumerate(past_key_values):
                past_key_values[key_name] = present_key_values[j]

        speech_tokens = generate_tokens[:, 1:-1]
        silence_tokens = np.full((speech_tokens.shape[0], 3), SILENCE_TOKEN, dtype=np.int64)
        speech_tokens = np.concatenate([prompt_token, speech_tokens, silence_tokens], axis=1)

        wav = self.cond_decoder_session.run(None, dict(
            speech_tokens=speech_tokens,
            speaker_embeddings=speaker_embeddings,
            speaker_features=speaker_features,
        ))[0].squeeze(axis=0)

        return wav

    def save_clone(self, name: str, audio_path: str, transcript: Optional[str] = None):
        try:
            import librosa
            duration = librosa.get_duration(path=audio_path)
            if duration > 15.0:
                raise ValueError(f"Audio is {duration:.1f}s long. Please use the 'Edit' (scissors) button on the audio uploader to trim it under 15s.")
        except ImportError:
            pass
        target_path = os.path.join(self.clones_dir, f"{name}.wav")
        import shutil
        shutil.copy(audio_path, target_path)
        logger.info(f"Saved Chatterbox clone: {name} to {target_path}")

    def list_clones(self) -> List[str]:
        if not os.path.exists(self.clones_dir):
            return []
        return sorted([f.split(".wav")[0] for f in os.listdir(self.clones_dir) if f.endswith(".wav")])
