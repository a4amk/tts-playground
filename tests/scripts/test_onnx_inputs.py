import onnxruntime
import sys

def print_inputs(path):
    print(f"--- {path} ---")
    try:
        sess = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'])
        for i in sess.get_inputs():
            print(f"  Input: {i.name}, Type: {i.type}, Shape: {i.shape}")
    except Exception as e:
        print("  Error:", e)

for p in [
    "models--ResembleAI--chatterbox-turbo-ONNX/snapshots/d21799bd0354adb85e348b8a0442a8405110a2cf/onnx/speech_encoder.onnx",
    "models--ResembleAI--chatterbox-turbo-ONNX/snapshots/d21799bd0354adb85e348b8a0442a8405110a2cf/onnx/language_model.onnx"
]:
    print_inputs("/home/ubuntu/.cache/huggingface/hub/" + p)
