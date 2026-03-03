import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Set up logging for configuration
logger = logging.getLogger(__name__)

# ------------------------------------------
# Hugging Face Configuration
# ------------------------------------------
# If LOCALIZE_MODELS is True, all model downloads are directed to 
# engine-specific folders in models_data/ instead of the global HF cache.
# This makes the app self-contained and portable.
LOCALIZE_MODELS = os.getenv("LOCALIZE_MODELS", "True").lower() in ("true", "1", "yes")

# HF_HOME is used for secondary libraries (like transformers) that don't
# allow passing a custom local_dir. We point it to models_data/huggingface.
HF_HOME = os.getenv("HF_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models_data", "huggingface")))
os.environ["HF_HOME"] = HF_HOME

# ------------------------------------------
# Hardware Configuration
# ------------------------------------------
USE_CPU_ONLY = os.getenv("USE_CPU_ONLY", "False").lower() in ("true", "1", "yes")

# If USE_CPU_ONLY is True, hide GPUs from all libraries (PyTorch, ONNX, etc.)
if USE_CPU_ONLY:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = "4" # Optimization for CPU parallelization

# ------------------------------------------
# Server Configuration
# ------------------------------------------
try:
    TTS_PORT = int(os.getenv("TTS_PORT", 7860))
except ValueError:
    logger.warning("Invalid TTS_PORT in .env, falling back to 7860")
    TTS_PORT = 7860

DEFAULT_TTS_ENGINE = os.getenv("DEFAULT_TTS_ENGINE", "kokoro")
ZIPVOICE_USE_ONNX = os.getenv("ZIPVOICE_USE_ONNX", "True").lower() in ("true", "1", "yes")

def get_device(fallback: str = "cpu") -> str:
    """
    Returns the appropriate hardware device based on the global USE_CPU_ONLY flag
    and system availability.
    """
    import torch
    if USE_CPU_ONLY:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return fallback
