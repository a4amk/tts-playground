import onnxruntime
from huggingface_hub import hf_hub_download

MODEL_ID = "ResembleAI/chatterbox-turbo-ONNX"

def load(dtype):
    print(f"Loading Chatterbox Turbo ONNX models (dtype={dtype})...")
    
    def download_model(name: str) -> str:
        filename = f"{name}{'' if dtype == 'fp32' else '_quantized' if dtype == 'q8' else f'_{dtype}'}.onnx"
        graph = hf_hub_download(MODEL_ID, subfolder="onnx", filename=filename)
        return graph
        
    conditional_decoder_path = download_model("conditional_decoder")
    speech_encoder_path = download_model("speech_encoder")
    embed_tokens_path = download_model("embed_tokens")
    language_model_path = download_model("language_model")

    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Try creating sessions
    speech_encoder_session = onnxruntime.InferenceSession(speech_encoder_path, opts, providers=['CPUExecutionProvider'])
    print(f"{dtype} parsed successfully!")

if __name__ == '__main__':
    for dt in ['q4f16', 'q8']:
        try:
            load(dt)
        except Exception as e:
            print(f"FAILED {dt}: {e}")
