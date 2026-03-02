import onnxruntime as ort
import numpy as np
from huggingface_hub import hf_hub_download

try:
    sess = ort.InferenceSession(hf_hub_download("hexgrad/styletts2", "4612a9dc0c0e142468f361e8e901bdccfdca45a2ae1145e5452bc98c7915302d.onnx"), providers=['CPUExecutionProvider'])
    
    valid = []
    for t in range(1, 400):
        try:
            sess.run(None, {"a": np.zeros((1, 80 * t), dtype=np.float32)})
            valid.append(t)
        except Exception as e:
            pass
            
    print("VALID:", valid)
except Exception as e:
    print("Outer Error:", e)
