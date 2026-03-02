import numpy as np
import re
from typing import AsyncGenerator, Tuple, Optional, List

class BaseTTS:
    """
    Abstract base class for all TTS models.
    """
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

    def get_available_voices(self) -> List[str]:
        """
        Return a list of local voice filenames or API voice names supported by this engine.
        """
        raise NotImplementedError

    def get_available_languages(self) -> List[str]:
        """
        Return a list of supported languages.
        """
        return ["auto"]
        
    async def generate_stream(self, text: str, voice: str, speed: float, **kwargs) -> AsyncGenerator[np.ndarray, None]:
        """
        Yield numpy float32 arrays chunk by chunk dynamically.
        """
        raise NotImplementedError
        yield # Makes it a generator
        
    def generate_batch(self, text: str, voice: str, speed: float, **kwargs) -> Optional[Tuple[int, np.ndarray]]:
        """
        Return a single full tuple representation like (24000, np.array_bytes)
        """
        raise NotImplementedError
