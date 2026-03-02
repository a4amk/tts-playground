import numpy as np
import asyncio
import scipy.signal
from typing import AsyncGenerator, Tuple, Optional, List
from ..base import BaseTTS
from .model import pocket_tts_model
from ..clones import clones_manager

class PocketTTSRuntime(BaseTTS):
    def get_available_voices(self) -> List[str]:
        predefined = pocket_tts_model.get_available_voices()
        user_clones = clones_manager.list_clones("pocket-tts")
        return predefined + user_clones

    async def generate_stream(self, text: str, voice: str, speed: float, **kwargs) -> AsyncGenerator[np.ndarray, None]:
        model = pocket_tts_model.get_model()
        
        # Check if it's a user clone
        clone_path = clones_manager.get_clone_path("pocket-tts", voice)
        if clone_path:
            if clone_path.endswith(".pt"):
                import torch
                prompt_state = torch.load(clone_path, map_location="cpu", weights_only=True)
            else:
                from pocket_tts.models.tts_model import _import_model_state
                prompt_state = _import_model_state(clone_path)
        else:
            prompt_state = pocket_tts_model.get_prompt_state(voice)
        
        # Override model temperature if needed
        user_temp = kwargs.get("temp", None)
        if user_temp is not None:
            model.temp = user_temp
        else:
            model.temp = 1.0 / max(0.1, speed)
        
        user_seed = kwargs.get("seed", 0)
        if user_seed > 0:
            import torch
            torch.manual_seed(user_seed)
        
        model.eval()
        
        # Unified splitting logic
        split_method = kwargs.get("split_method", "Sentences")
        custom_regex = kwargs.get("custom_regex", "")
        chunks = self.split_text(text, split_method, custom_regex)
        
        source_sr = model.config.mimi.sample_rate
        target_sr = 24000
        loop = asyncio.get_event_loop()

        for chunk_text in chunks:
            if not chunk_text.strip(): continue
            
            import threading
            import queue
            q = queue.Queue()
            
            # Use model's generator for this chunk
            audio_gen = model.generate_audio_stream(
                model_state=prompt_state,
                text_to_generate=chunk_text,
                copy_state=True # Keep original prompt state intact
            )

            def tts_worker(generator, out_q):
                 try:
                      for c in generator:
                           out_q.put(("data", c))
                 except Exception as e:
                      out_q.put(("error", e))
                 finally:
                      out_q.put(("done", None))

            worker_thread = threading.Thread(target=tts_worker, args=(audio_gen, q))
            worker_thread.start()
            
            try:
                 while True:
                      msg_type, chunk = await loop.run_in_executor(None, q.get)
                      if msg_type == "done": break
                      elif msg_type == "error":
                           print(f"Pocket-TTS stream error: {chunk}")
                           break
                      elif msg_type == "data" and chunk is not None:
                           audio_np = chunk.detach().cpu().numpy().flatten()
                           if source_sr != target_sr:
                                num_samples = int(len(audio_np) * target_sr / source_sr)
                                audio_np = scipy.signal.resample(audio_np, num_samples)
                           yield audio_np.astype(np.float32)
            except asyncio.CancelledError:
                 break
            finally:
                 worker_thread.join(timeout=1.0) # Graceful wait

    def generate_batch(self, text: str, voice: str, speed: float, **kwargs) -> Optional[Tuple[int, np.ndarray]]:
        # Map voice to prompt state
        model = pocket_tts_model.get_model()
        clone_path = clones_manager.get_clone_path("pocket-tts", voice)
        if clone_path:
            if clone_path.endswith(".pt"):
                import torch
                prompt_state = torch.load(clone_path, map_location="cpu", weights_only=True)
            else:
                from pocket_tts.models.tts_model import _import_model_state
                prompt_state = _import_model_state(clone_path)
        else:
            prompt_state = pocket_tts_model.get_prompt_state(voice)

        # Basic collection
        chunks = []
        source_sr = model.config.mimi.sample_rate
        
        # User internal splitting or manual? Let's use internal for batch
        for chunk in model.generate_audio_stream(model_state=prompt_state, text_to_generate=text):
             chunks.append(chunk.detach().cpu().numpy().flatten())
        
        if not chunks:
             return None
             
        full_audio = np.concatenate(chunks)
        return source_sr, (full_audio * 32767).astype(np.int16)

pocket_tts_runtime = PocketTTSRuntime()
