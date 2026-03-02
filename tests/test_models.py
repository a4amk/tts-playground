import pytest
from app.engines.registry import models
from app.engines.base import BaseTTS

def test_registry_contains_kokoro():
    assert "kokoro" in models
    assert isinstance(models["kokoro"], BaseTTS)

def test_kokoro_voices():
    engine = models["kokoro"]
    voices = engine.get_available_voices()
    assert isinstance(voices, list)
    assert len(voices) > 0
    assert "af_heart" in voices or len(voices) > 0

def test_kokoro_batch_generation():
    engine = models["kokoro"]
    # Test with a very short string to keep it fast
    result = engine.generate_batch(text="Hello", voice="af_heart", speed=1.0)
    assert result is not None
    rate, audio = result
    assert rate == 24000
    assert audio.dtype == "int16"
    assert len(audio) > 0
