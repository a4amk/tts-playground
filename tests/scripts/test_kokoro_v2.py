import onnxruntime as ort
import numpy as np
import os

def test():
    # Load kokoro model
    model_path = "/home/ubuntu/my-apps/tts-playground/models_data/kokoro/kokoro-v0_19.onnx"
    if not os.path.exists(model_path):
        print("Kokoro not found at expected path.")
        return
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # Simple input for kokoro
    tokens = np.array([[10, 20, 30]], dtype=np.int64)
    style = np.zeros((1, 256), dtype=np.float32)
    speed = np.array([1.0], dtype=np.float32)
    
    # Try multiple run names, kokoro v0.19 usually:
    # input name: "tokens", "style", "speed"
    out = sess.run(None, {"tokens": tokens, "style": style, "speed": speed})[0]
    print(f"Kokoro Out: Max {np.max(np.abs(out))}, Mean {np.mean(np.abs(out))}")

test()
