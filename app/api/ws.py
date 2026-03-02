import json
import asyncio
import re
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ..engines.registry import models

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
                engine = models.get(model_id)
                if not engine:
                    await websocket.send_bytes(b'')
                    return
                
                # Single-pass legacy mode
                generator = engine.generate_stream(
                    text=params["text"],
                    voice=params.get("voice", "af_heart"), 
                    speed=float(params.get("speed", 1.0)), 
                    lang=params.get("lang", "a"),
                    split_choice=params.get("split_choice", "Both (Newlines & Sentences)"),
                    custom_regex=params.get("custom_regex", r'\n+'),
                    temp=float(params.get("temp", 0.7)),
                    top_k=int(params.get("top_k", 50)),
                    top_p=float(params.get("top_p", 0.9)),
                    rep_pen=float(params.get("rep_pen", 1.0)),
                    seed=int(params.get("seed", 0)),
                    cfg=float(params.get("cfg", 0.5)),
                    exaggeration=float(params.get("exaggeration", 0.5))
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
                
                engine = models.get(params.get("model", "kokoro"))
                for sentence in chunks:
                    if not sentence.strip(): continue
                    print(f"Synthesizing sentence: {sentence}")
                    generator = engine.generate_stream(
                        text=sentence,
                        voice=params.get("voice"),
                        speed=float(params.get("speed", 1.0)),
                        lang=params.get("lang", "en"),
                        temp=float(params.get("temp", 0.7)),
                        top_k=int(params.get("top_k", 50)),
                        top_p=float(params.get("top_p", 0.9)),
                        rep_pen=float(params.get("rep_pen", 1.0)),
                        seed=int(params.get("seed", 0)),
                        cfg=float(params.get("cfg", 0.5)),
                        exaggeration=float(params.get("exaggeration", 0.5))
                    )
                    async for chunk in generator:
                        if chunk is not None and len(chunk) > 0:
                            await websocket.send_bytes(chunk.tobytes())
                            
            if params and op == "flush":
                # Finalize any remaining text
                if text_buffer.strip():
                    engine = models.get(params.get("model", "kokoro"))
                    print(f"Flushing remaining text: {text_buffer}")
                    generator = engine.generate_stream(
                        text=text_buffer,
                        voice=params.get("voice"),
                        speed=float(params.get("speed", 1.0)),
                        lang=params.get("lang", "en"),
                        temp=float(params.get("temp", 0.7)),
                        top_k=int(params.get("top_k", 50)),
                        top_p=float(params.get("top_p", 0.9)),
                        rep_pen=float(params.get("rep_pen", 1.0)),
                        seed=int(params.get("seed", 0)),
                        cfg=float(params.get("cfg", 0.5)),
                        exaggeration=float(params.get("exaggeration", 0.5))
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
