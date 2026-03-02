import pytest
from fastapi.testclient import TestClient
from main import app
import json

client = TestClient(app)

def test_gradio_home_renders():
    # Verify the Gradio app is mounted and responding at root
    response = client.get("/")
    assert response.status_code == 200
    assert "gradio" in response.text.lower()

def test_websocket_stream_connect():
    # WebSocket testing using TestClient
    with client.websocket_connect("/ws/stream") as websocket:
        # Send a valid payload for the registry
        payload = {
            "text": "Unit Test Stream",
            "model": "kokoro",
            "voice": "af_heart",
            "lang": "a",
            "speed": 1.0,
            "split_choice": "Both (Newlines & Sentences)"
        }
        websocket.send_text(json.dumps(payload))
        
        # Receive data - expect some binary chunks
        data = websocket.receive_bytes()
        assert len(data) > 0 # Initial audio data
        
        # The engine eventually sends an empty byte string to signal finish
        # (Though we might receive multiple chunks first depending on speed)
        # We just want to check that it is indeed streaming
        assert isinstance(data, bytes)
