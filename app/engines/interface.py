import numpy as np
import re
from typing import AsyncGenerator, Tuple, Optional, List, Dict, Any
from abc import ABC, abstractmethod

class TTSPlugin(ABC):
    """
    Unified interface for all TTS Engines in the pluggable architecture.
    """
    
    # --- Metadata & Discovery ---
    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for the engine (e.g., 'kokoro_onnx')."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name for the UI."""
        pass

    # --- UI Configuration ---
    @abstractmethod
    def get_ui_config(self) -> Dict[str, Any]:
        """
        Returns default slider values and constraints.
        Example: {'temperature': 0.7, 'top_p': 1.0, 'repetition_penalty': 1.1}
        """
        pass

    @abstractmethod
    def get_cloning_config(self) -> Dict[str, Any]:
        """
        Returns requirements for voice cloning.
        Example: {'requires_transcript': True, 'max_chars': 200}
        """
        pass

    def get_extra_controls(self) -> List[Dict[str, Any]]:
        """
        Returns a list of extra UI controls for this engine.
        Example: [{"id": "use_onnx", "type": "checkbox", "label": "Use ONNX", "default": True}]
        Supported types: 'checkbox', 'dropdown', 'slider'
        """
        return []

    @abstractmethod
    def get_standard_controls(self) -> List[Dict[str, Any]]:
        """
        Returns metadata for standard controls (speed, temp, top_k, top_p, rep_pen, seed, cfg, exaggeration).
        If a control is not returned, it will be hidden in the UI.
        Format: [{"id": "temp", "label": "Temperature", "info": "...", "min": 0.1, "max": 2.0, "step": 0.1, "default": 0.7}]
        """
        pass

    @abstractmethod
    def get_variants(self) -> List[Dict[str, Any]]:
        """
        Returns a list of model variants (e.g. runtimes, quantization levels).
        Example: [{"id": "fp32", "label": "FP32", "default": True}, {"id": "int8", "label": "INT8", "default": False}]
        """
        pass

    # --- Capabilities ---
    @abstractmethod
    def get_available_voices(self) -> List[str]:
        """List of supported speaker/voice IDs."""
        pass

    @abstractmethod
    def get_available_languages(self) -> List[str]:
        """List of supported ISO language codes."""
        return ["en"]

    # --- Lifecycle ---
    @abstractmethod
    def is_installed(self) -> bool:
        """Checks if required python dependencies or system binaries are present."""
        return True

    @abstractmethod
    def install_dependencies(self):
        """Triggers pip/apt installation logic for the engine."""
        pass

    @abstractmethod
    def load(self, variant: Optional[str] = None):
        """Lazy loader for model weights and inference sessions."""
        pass

    # --- Synthesis ---
    @abstractmethod
    async def generate_stream(self, text: str, voice: str, speed: float, variant: Optional[str] = None, **kwargs) -> AsyncGenerator[np.ndarray, None]:
        """Yields raw float32 audio chunks."""
        pass

    @abstractmethod
    def generate_batch(self, text: str, voice: str, speed: float, variant: Optional[str] = None, **kwargs) -> Optional[Tuple[int, np.ndarray]]:
        """Returns (sample_rate, int16_audio_data) for file downloads."""
        pass

    # --- Cloning logic ---
    @abstractmethod
    def save_clone(self, name: str, audio_path: str, transcript: Optional[str] = None):
        """Serializes a new voice clone to the plugin's local storage."""
        pass

    @abstractmethod
    def list_clones(self) -> List[str]:
        """Lists available custom voices for this specific engine."""
        return []

    # --- Shared Utilities ---
    split_strategies = {
        "Both (Newlines & Sentences)": r'\n+|(?<=[.!?])\s+',
        "Sentences (Punctuation)": r'(?<=[.!?])\s+',
        "Paragraphs (Newlines)": r'\n+',
        "Words (Spaces)": r'\s+',
        "No Splitting (Single Pass)": r'^$',
        "Custom Regex": "custom"
    }

    def _resolve_split_pattern(self, split_choice: str, custom_regex: str) -> str:
        if split_choice == "Custom Regex":
            return custom_regex
        return self.split_strategies.get(split_choice, r'\n+')

    def split_text(self, text: str, split_choice: str, custom_regex: str) -> List[str]:
        pattern = self._resolve_split_pattern(split_choice, custom_regex)
        if pattern == r'^$':
            return [text]
        return [s.strip() for s in re.split(pattern, text) if s.strip()]
