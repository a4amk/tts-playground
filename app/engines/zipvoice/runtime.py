import numpy as np
import os
import asyncio
import datetime as dt
from typing import AsyncGenerator, Tuple, Optional, List
from ..base import BaseTTS
from .model import zipvoice_model, zipvoice_dir
import sys

# ZipVoice uses a combination of audio features and reference text for cloning.
import re
from ..clones import clones_manager

if zipvoice_dir not in sys.path:
    sys.path.append(zipvoice_dir)

from zipvoice.utils.infer import (
    add_punctuation,
    batchify_tokens,
    chunk_tokens_punctuation,
    cross_fade_concat,
    load_prompt_wav,
    remove_silence,
    rms_norm,
)

class ZipVoiceRuntime(BaseTTS):
    def __init__(self):
        self.prompt_cache = {}
    def get_available_voices(self) -> List[str]:
        predefined = zipvoice_model.get_available_voices()
        user_clones = clones_manager.list_clones("zipvoice")
        return predefined + user_clones

    async def generate_stream(self, text: str, voice: str, speed: float, **kwargs) -> AsyncGenerator[np.ndarray, None]:
        zipvoice_model.load_model()
        model = zipvoice_model.model
        vocoder = zipvoice_model.vocoder
        tokenizer = zipvoice_model.tokenizer
        feature_extractor = zipvoice_model.feature_extractor
        device = zipvoice_model.device
        
        # Check if it's a user clone
        ref_path = clones_manager.get_clone_path("zipvoice", voice)
        if not ref_path:
            ref_path = os.path.join(zipvoice_model.ref_dir, voice)
        
        txt_path = os.path.splitext(ref_path)[0] + ".txt"
        
        prompt_text = ""
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()
                
        def _generate():
            import torch
            import torchaudio
            import datetime as dt
            import re
            
            sampling_rate = 24000
            feat_scale = 0.1
            target_rms = 0.1
            
            # Cache key for prompt features
            cache_key = (voice, prompt_text)
            if cache_key in self.prompt_cache:
                prompt_features, prompt_rms = self.prompt_cache[cache_key]
                print(f"ZipVoice: Using cached prompt for {voice}")
            else:
                prompt_wav = load_prompt_wav(ref_path, sampling_rate=sampling_rate)
                prompt_wav = remove_silence(prompt_wav, sampling_rate, only_edge=False, trail_sil=200)
                prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)
                
                prompt_features = feature_extractor.extract(prompt_wav, sampling_rate=sampling_rate).to(device)
                prompt_features = prompt_features.unsqueeze(0) * feat_scale
                self.prompt_cache[cache_key] = (prompt_features, prompt_rms)
                print(f"ZipVoice: Processed and cached prompt for {voice}")
            
            pt = add_punctuation(prompt_text)
            prompt_tokens_str = tokenizer.texts_to_tokens([pt])[0]
            prompt_tokens = tokenizer.tokens_to_token_ids([prompt_tokens_str])

            # Use chunking strategy
            split_choice = kwargs.get("split_choice", "Sentences (Punctuation)")
            custom_regex = kwargs.get("custom_regex", r'\n+')
            chunk_texts = self.split_text(text, split_choice, custom_regex)
            
            print(f"ZipVoice: Split text into {len(chunk_texts)} chunks using {split_choice}.")
            
            for i, chunk_text in enumerate(chunk_texts):
                start_chunk = dt.datetime.now()
                
                # ZipVoice internal tokenization for this chunk
                t = add_punctuation(chunk_text)
                tokens_str = tokenizer.texts_to_tokens([t])[0]
                # ZipVoice still likes internally chunked tokens if they are very long
                # but we respect the outer chunking. If one outer chunk is still too long, 
                # we'll let ZipVoice's internal chunking handle it at the token level (max 50 tokens)
                chunked_token_sequences_str = chunk_tokens_punctuation(tokens_str, max_tokens=50)
                
                for j, sub_tokens_str in enumerate(chunked_token_sequences_str):
                    sub_token_ids = tokenizer.tokens_to_token_ids([sub_tokens_str])[0]
                    
                    with torch.inference_mode():
                        batch_prompt_features_lens = torch.full((1,), prompt_features.size(1), device=device)
                        
                        (
                            pred_features,
                            pred_features_lens,
                            _, _
                        ) = model.sample(
                            tokens=[sub_token_ids],
                            prompt_tokens=prompt_tokens,
                            prompt_features=prompt_features,
                            prompt_features_lens=batch_prompt_features_lens,
                            speed=speed,
                            t_shift=0.5,
                            duration="predict",
                            num_step=8 if zipvoice_model.model_name == "zipvoice_distill" else 16,
                            guidance_scale=3.0 if zipvoice_model.model_name == "zipvoice_distill" else 1.0,
                        )
                        
                        pred_features = pred_features.permute(0, 2, 1) / feat_scale
                        wav = vocoder.decode(pred_features[0][None, :, : pred_features_lens[0]]).squeeze(1).clamp(-1, 1)
                        
                        if prompt_rms < target_rms:
                            wav = wav * prompt_rms / target_rms
                        
                        duration = (dt.datetime.now() - start_chunk).total_seconds()
                        audio_duration = wav.shape[-1] / sampling_rate
                        print(f"ZipVoice: Chunk {i+1}.{j+1} generated in {duration:.2f}s (audio length: {audio_duration:.2f}s, RTF: {duration/audio_duration:.2f})")
                        
                        yield wav.cpu().numpy().flatten().astype(np.float32)
        
        loop = asyncio.get_event_loop()
        
        import threading
        import queue
        q = queue.Queue()
        
        def tts_worker(out_q):
             try:
                  for chunk in _generate():
                       out_q.put(("data", chunk))
             except Exception as e:
                  import traceback
                  traceback.print_exc()
                  out_q.put(("error", str(e)))
             finally:
                  out_q.put(("done", None))

        worker_thread = threading.Thread(target=tts_worker, args=(q,))
        worker_thread.start()
        
        while True:
             msg_type, chunk = await loop.run_in_executor(None, q.get)
             
             if msg_type == "done":
                  break
             elif msg_type == "error":
                  print(f"ZipVoice stream error: {chunk}")
                  break
             elif msg_type == "data" and chunk is not None:
                  yield chunk
             
             await asyncio.sleep(0.001)

    def generate_batch(self, text: str, voice: str, speed: float, **kwargs) -> Optional[Tuple[int, np.ndarray]]:
         # Fallback to sync run
         return None

zipvoice_runtime = ZipVoiceRuntime()
