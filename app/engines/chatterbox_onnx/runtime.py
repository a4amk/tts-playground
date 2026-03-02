import os
import asyncio
import numpy as np
from typing import AsyncGenerator, Tuple, Optional, List
import librosa

from ..base import BaseTTS
from ..clones import StoredVoicesManager
from .model import get_model, SAMPLE_RATE, START_SPEECH_TOKEN, STOP_SPEECH_TOKEN, SILENCE_TOKEN, NUM_KV_HEADS, HEAD_DIM, RepetitionPenaltyLogitsProcessor

clones_manager = StoredVoicesManager()

class ChatterboxONNXRuntime(BaseTTS):
    def get_available_voices(self) -> List[str]:
        # Chatterbox is built primarily around voice cloning
        clones = clones_manager.list_clones("chatterbox-turbo-onnx")
        voices = ["default"] + clones
        return voices

    def get_available_languages(self) -> List[str]:
        # Chatterbox Turbo is essentially English
        return ["en"]

    def _prepare_audio_values(self, voice: str) -> np.ndarray:
        voice_path = None
        if voice != "default":
            # Check user clone
            clone_file = clones_manager.get_clone_path("chatterbox-turbo-onnx", voice)
            if clone_file and os.path.exists(clone_file):
                voice_path = clone_file
                
        # Default reference voice fallback (Genie's base dummy config)
        if not voice_path:
            voice_path = "/home/ubuntu/my-apps/tts-playground/models_data/chatterbox_onnx/female_shadowheart4.flac"
            
        audio_values, _ = librosa.load(voice_path, sr=SAMPLE_RATE)
        return audio_values[np.newaxis, :].astype(np.float32)

    async def generate_stream(self, text: str, voice: str, speed: float, **kwargs) -> AsyncGenerator[np.ndarray, None]:
        # Note: Chatterbox ONNX does not natively support continuous sub-word stream playback 
        # because the decode step requires all completed tokens to be generated.
        # We chunk by sentences and yield each completed sentence's full waveform instead.
        
        split_choice = kwargs.get("split_choice", "Both (Newlines & Sentences)")
        custom_regex = kwargs.get("custom_regex", r'\n+')
        chunks = self.split_text(text, split_choice, custom_regex)
        
        audio_values = self._prepare_audio_values(voice)
        
        for chunk_text in chunks:
            if not chunk_text.strip(): continue
            audio = await asyncio.to_thread(self._generate_single, chunk_text, audio_values, **kwargs)
            if audio is not None:
                yield audio
                
    def generate_batch(self, text: str, voice: str, speed: float, **kwargs) -> Optional[Tuple[int, np.ndarray]]:
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
            return SAMPLE_RATE, np.concatenate(audio_chunks)
        return None

    def _generate_single(self, text: str, audio_values: np.ndarray, **kwargs) -> Optional[np.ndarray]:
        model = get_model()
        
        max_new_tokens = 1000
        repetition_penalty = float(kwargs.get("rep_pen", 1.2))
        top_k = int(kwargs.get("top_k", 50))
        temp = float(kwargs.get("temp", 0.7))
        speed = float(kwargs.get("speed", 1.0))
        
        input_ids = model.tokenizer(text, return_tensors="np")["input_ids"].astype(np.int64)
        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)
        generate_tokens = np.array([[START_SPEECH_TOKEN]], dtype=np.int64)
        
        # Generation loop
        for i in range(max_new_tokens):
            inputs_embeds = model.embed_tokens_session.run(None, {"input_ids": input_ids})[0]

            if i == 0:
                ort_speech_encoder_input = {"audio_values": audio_values}
                cond_emb, prompt_token, speaker_embeddings, speaker_features = model.speech_encoder_session.run(None, ort_speech_encoder_input)
                inputs_embeds = np.concatenate((cond_emb, inputs_embeds), axis=1)

                batch_size, seq_len, _ = inputs_embeds.shape
                past_key_values = {
                    i_in.name: np.zeros([batch_size, NUM_KV_HEADS, 0, HEAD_DIM], dtype=np.float16 if i_in.type == 'tensor(float16)' else np.float32)
                    for i_in in model.language_model_session.get_inputs()
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
            
            output = model.language_model_session.run(None, run_dict)
            logits = output[0]
            present_key_values = output[1:]

            logits = logits[:, -1, :]
            next_token_logits = repetition_penalty_processor(generate_tokens, logits)

            # Deterministic/Greedy fallback for robust ONNX execution
            if temp == 0:
                 new_input_ids = np.argmax(next_token_logits, axis=-1, keepdims=True).astype(np.int64)
            else:
                 # Sample if temp > 0
                 probs = np.exp(next_token_logits / max(temp, 0.1)) / np.sum(np.exp(next_token_logits / max(temp, 0.1)), axis=-1, keepdims=True)
                 new_input_ids = np.array([[np.random.choice(len(probs[0]), p=probs[0])]], dtype=np.int64)

            generate_tokens = np.concatenate((generate_tokens, new_input_ids), axis=-1)
            
            if (new_input_ids.flatten() == STOP_SPEECH_TOKEN).all():
                break

            input_ids = new_input_ids
            attention_mask = np.concatenate([attention_mask, np.ones((batch_size, 1), dtype=np.int64)], axis=1)
            position_ids = position_ids[:, -1:] + 1
            for j, key in enumerate(past_key_values):
                past_key_values[key] = present_key_values[j]

        # Decode audio
        speech_tokens = generate_tokens[:, 1:-1]
        silence_tokens = np.full((speech_tokens.shape[0], 3), SILENCE_TOKEN, dtype=np.int64)
        speech_tokens = np.concatenate([prompt_token, speech_tokens, silence_tokens], axis=1)

        wav = model.cond_decoder_session.run(None, dict(
            speech_tokens=speech_tokens,
            speaker_embeddings=speaker_embeddings,
            speaker_features=speaker_features,
        ))[0].squeeze(axis=0)

        if speed != 1.0:
            wav = librosa.effects.time_stretch(wav, rate=speed)
            
        return wav
