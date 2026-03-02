import uvicorn
from fastapi import FastAPI
import gradio as gr
import torch

from app.config import TTS_PORT

# Multi-engine compatibility patches
try:
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    if not hasattr(EspeakWrapper, 'set_data_path'):
        EspeakWrapper.set_data_path = staticmethod(lambda path: None)
    if not hasattr(EspeakWrapper, 'set_library'):
        EspeakWrapper.set_library = staticmethod(lambda path: None)
except ImportError:
    pass

from app.api.ws import ws_router
from app.ui.gradio_app import create_blocks

app = FastAPI()

# Map the websocket routing namespace manually configured behind our fastapi logic
app.include_router(ws_router)

# Fetch our Gradio front end class block and tie it cleanly to our API framework
blocks_app = create_blocks()

# Ensure that the root directory runs gradio by default over standard fastapi index calls
app = gr.mount_gradio_app(app, blocks_app, path="/")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=TTS_PORT, reload=False)
