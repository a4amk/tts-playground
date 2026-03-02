import json
import asyncio
import re
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ..engines.manager import plugin_manager

ws_router = APIRouter()

def get_sentence_chunks(text, buffer):
    """
    Chunks text at sentence boundaries. Returns (list_of_chunks, remaining_buffer).
    """
    full_text = buffer + text
    # Split on punctuation followed by space or newline
    pattern = r'(?<=[.!?])(?:\s+|\n+)'
    chunks = re.split(pattern, full_text)
    
    if len(chunks) > 1:
        # The last element might be an incomplete sentence
        return chunks[:-1], chunks[-1]
    else:
        return [], full_text

@ws_router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected!")
    
    params = None
    text_buffer = ""
    
    try:
        while True:
            # Wait for the first message to get config
            msg = await websocket.receive_text()
            data = json.loads(msg)
            
            # If it's the old format (full text in first msg), handle it and exit after
            if "text" in data and "model" in data and params is None:
                params = data
                model_id = params.get("model", "kokoro")
                engine = plugin_manager.get_plugin(model_id)
                if not engine:
                    await websocket.send_bytes(b'')
                    return
                
                # Map extras back to kwargs using engine definitions
                extra_definitions = engine.get_extra_controls()
                extra_kwargs = {}
                for i, ctrl in enumerate(extra_definitions):
                    key = f"extra_{i}"
                    if key in params:
                        # Defensive check: avoid overriding core arguments
                        if ctrl["id"] not in ["text", "model", "voice", "lang", "speed", "split_choice", "custom_regex", "temp", "top_k", "top_p", "rep_pen", "seed", "cfg", "exaggeration"]:
                            extra_kwargs[ctrl["id"]] = params[key]

                # Single-pass legacy mode
                generator = engine.generate_stream(
                    text=params["text"],
                    voice=params.get("voice", "af_heart"), 
                    speed=float(params.get("speed", 1.0)), 
                    lang=params.get("lang", "a"),
                    variant=params.get("variant"),
                    split_choice=params.get("split_choice", "Both (Newlines & Sentences)"),
                    custom_regex=params.get("custom_regex", r'\n+'),
                    temp=float(params.get("temp", 0.7)),
                    top_k=int(params.get("top_k", 50)),
                    top_p=float(params.get("top_p", 0.9)),
                    rep_pen=float(params.get("rep_pen", 1.0)),
                    seed=int(params.get("seed", 0)),
                    cfg=float(params.get("cfg", 0.5)),
                    exaggeration=float(params.get("exaggeration", 0.5)),
                    **extra_kwargs
                )
                async for chunk in generator:
                    if chunk is not None and len(chunk) > 0:
                        await websocket.send_bytes(chunk.tobytes())
                await websocket.send_bytes(b'')
                return
            
            # New Incremental Protocol: 
            # 1. { "op": "start", "model": ..., "voice": ... }
            # 2. { "op": "text", "value": ... }
            # 3. { "op": "flush" } 
            
            op = data.get("op")
            if op == "start":
                params = data
                print(f"Streaming mode started for model: {params.get('model')}")
                continue
                
            if params and op == "text":
                new_text = data.get("value", "")
                chunks, text_buffer = get_sentence_chunks(new_text, text_buffer)
                
                engine = plugin_manager.get_plugin(params.get("model", "kokoro"))
                for sentence in chunks:
                    if not sentence.strip(): continue
                    print(f"Synthesizing sentence: {sentence}")
                    # Map extras
                    extra_definitions = engine.get_extra_controls()
                    extra_kwargs = {}
                    for i, ctrl in enumerate(extra_definitions):
                        key = f"extra_{i}"
                        if key in params:
                            # Defensive check: avoid overriding core arguments
                            if ctrl["id"] not in ["text", "model", "voice", "lang", "speed", "split_choice", "custom_regex", "temp", "top_k", "top_p", "rep_pen", "seed", "cfg", "exaggeration"]:
                                extra_kwargs[ctrl["id"]] = params[key]

                    generator = engine.generate_stream(
                        text=sentence,
                        voice=params.get("voice"),
                        speed=float(params.get("speed", 1.0)),
                        lang=params.get("lang", "en"),
                        variant=params.get("variant"),
                        temp=float(params.get("temp", 0.7)),
                        top_k=int(params.get("top_k", 50)),
                        top_p=float(params.get("top_p", 0.9)),
                        rep_pen=float(params.get("rep_pen", 1.0)),
                        seed=int(params.get("seed", 0)),
                        cfg=float(params.get("cfg", 0.5)),
                        exaggeration=float(params.get("exaggeration", 0.5)),
                        **extra_kwargs
                    )
                    async for chunk in generator:
                        if chunk is not None and len(chunk) > 0:
                            await websocket.send_bytes(chunk.tobytes())
                            
            if params and op == "flush":
                # Finalize any remaining text
                if text_buffer.strip():
                    engine = plugin_manager.get_plugin(params.get("model", "kokoro"))
                    print(f"Flushing remaining text: {text_buffer}")
                    # Map extras
                    extra_definitions = engine.get_extra_controls()
                    extra_kwargs = {}
                    for i, ctrl in enumerate(extra_definitions):
                        key = f"extra_{i}"
                        if key in params:
                            # Defensive check: avoid overriding core arguments
                            if ctrl["id"] not in ["text", "model", "voice", "lang", "speed", "split_choice", "custom_regex", "temp", "top_k", "top_p", "rep_pen", "seed", "cfg", "exaggeration"]:
                                extra_kwargs[ctrl["id"]] = params[key]

                    generator = engine.generate_stream(
                        text=text_buffer,
                        voice=params.get("voice"),
                        speed=float(params.get("speed", 1.0)),
                        lang=params.get("lang", "en"),
                        variant=params.get("variant"),
                        temp=float(params.get("temp", 0.7)),
                        top_k=int(params.get("top_k", 50)),
                        top_p=float(params.get("top_p", 0.9)),
                        rep_pen=float(params.get("rep_pen", 1.0)),
                        seed=int(params.get("seed", 0)),
                        cfg=float(params.get("cfg", 0.5)),
                        exaggeration=float(params.get("exaggeration", 0.5)),
                        **extra_kwargs
                    )
                    async for chunk in generator:
                        if chunk is not None and len(chunk) > 0:
                            await websocket.send_bytes(chunk.tobytes())
                    text_buffer = ""
                await websocket.send_bytes(b'')
                print("WebSocket stream flushed.")
                
            if op == "stop":
                break

    except WebSocketDisconnect:
        print("Client disconnected websocket.")
    except Exception as e:
        print(f"WebSocket Error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass
