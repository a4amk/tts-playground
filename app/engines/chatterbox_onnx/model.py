import os
import onnxruntime
import numpy as np
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

MODEL_ID = "ResembleAI/chatterbox-turbo-ONNX"
SAMPLE_RATE = 24000
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562
SILENCE_TOKEN = 4299
NUM_KV_HEADS = 16
HEAD_DIM = 64

class ChatterboxONNXModel:
    def __init__(self):
        self.speech_encoder_session = None
        self.embed_tokens_session = None
        self.language_model_session = None
        self.cond_decoder_session = None
        self.tokenizer = None
        self.is_loaded = False
        
    def load(self, dtype="fp32"):
        if self.is_loaded: return
        
        print(f"Loading Chatterbox Turbo ONNX models (dtype={dtype})...")
        conditional_decoder_path = self.download_model("conditional_decoder", dtype=dtype)
        speech_encoder_path = self.download_model("speech_encoder", dtype=dtype)
        embed_tokens_path = self.download_model("embed_tokens", dtype=dtype)
        language_model_path = self.download_model("language_model", dtype=dtype)

        opts = onnxruntime.SessionOptions()
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.speech_encoder_session = onnxruntime.InferenceSession(speech_encoder_path, opts, providers=['CPUExecutionProvider'])
        self.embed_tokens_session = onnxruntime.InferenceSession(embed_tokens_path, opts, providers=['CPUExecutionProvider'])
        self.language_model_session = onnxruntime.InferenceSession(language_model_path, opts, providers=['CPUExecutionProvider'])
        self.cond_decoder_session = onnxruntime.InferenceSession(conditional_decoder_path, opts, providers=['CPUExecutionProvider'])
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.is_loaded = True

    def download_model(self, name: str, dtype: str = "fp32") -> str:
        filename = f"{name}{'' if dtype == 'fp32' else '_quantized' if dtype == 'q8' else f'_{dtype}'}.onnx"
        graph = hf_hub_download(MODEL_ID, subfolder="onnx", filename=filename)
        hf_hub_download(MODEL_ID, subfolder="onnx", filename=f"{filename}_data")
        return graph

_instance = ChatterboxONNXModel()

def get_model():
    _instance.load()
    return _instance

class RepetitionPenaltyLogitsProcessor:
    def __init__(self, penalty: float):
        penalty = float(penalty)
        if not (penalty > 0):
            raise ValueError(f"`penalty` must be a strictly positive float, but is {penalty}")
        self.penalty = penalty

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        score = np.take_along_axis(scores, input_ids, axis=1)
        score = np.where(score < 0, score * self.penalty, score / self.penalty)
        scores_processed = scores.copy()
        np.put_along_axis(scores_processed, input_ids, score, axis=1)
        return scores_processed
