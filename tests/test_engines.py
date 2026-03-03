import pytest
import asyncio
import numpy as np
import os
import sys

# Ensure app is in path
sys.path.append(os.getcwd())

from app.engines.manager import plugin_manager

@pytest.mark.asyncio
async def test_all_engines_discovery():
    """Verify all expected engines are discovered."""
    plugin_manager.discover_plugins()
    ids = plugin_manager.get_all_ids()
    print(f"Discovered engines: {ids}")
    # Kokoro and ZipVoice should be here now
    assert "kokoro" in ids
    assert "zipvoice" in ids
    assert "piper" in ids
    assert "chatterbox_onnx" in ids
    assert "pocket_tts" in ids

@pytest.mark.parametrize("engine_id", plugin_manager.get_all_ids() if plugin_manager.get_all_ids() else ["kokoro", "zipvoice", "piper", "chatterbox_onnx", "pocket_tts"])
@pytest.mark.asyncio
async def test_engine_initialization(engine_id):
    """Test that each engine can load its models without crashing."""
    plugin_manager.discover_plugins()
    engine = plugin_manager.get_plugin(engine_id)
    assert engine is not None, f"Engine {engine_id} not found"
    
    print(f"Testing initialization for: {engine.display_name}")
    # load() should be idempotent and not crash
    engine.load()
    
    voices = engine.get_available_voices()
    assert len(voices) > 0, f"Engine {engine_id} has no voices"
    print(f"  Available voices: {len(voices)}")

@pytest.mark.parametrize("engine_id", ["pocket_tts", "kokoro", "piper"]) # Test a subset first to save time/memory
@pytest.mark.asyncio
async def test_engine_generation_batch(engine_id):
    """Test batch generation for engines."""
    plugin_manager.discover_plugins()
    engine = plugin_manager.get_plugin(engine_id)
    if not engine: pytest.skip(f"Engine {engine_id} not available")
    
    voices = engine.get_available_voices()
    voice = voices[0]
    
    print(f"Testing batch generation for {engine_id} with voice {voice}")
    result = await asyncio.to_thread(engine.generate_batch, "Hello testing.", voice, speed=1.0)
    
    assert result is not None, f"Batch generation failed for {engine_id}"
    sr, audio = result
    assert sr > 0
    assert len(audio) > 0
    assert audio.dtype == np.int16

@pytest.mark.parametrize("engine_id", ["kokoro", "piper"]) 
@pytest.mark.asyncio
async def test_engine_generation_stream(engine_id):
    """Test streaming generation for engines."""
    plugin_manager.discover_plugins()
    engine = plugin_manager.get_plugin(engine_id)
    if not engine: pytest.skip(f"Engine {engine_id} not available")
    
    voices = engine.get_available_voices()
    voice = voices[0]
    
    print(f"Testing streaming generation for {engine_id} with voice {voice}")
    generator = engine.generate_stream("Hello testing stream.", voice, speed=1.0)
    
    chunks = []
    async for chunk in generator:
        if chunk is not None and len(chunk) > 0:
            chunks.append(chunk)
            break # Just need to verify the first chunk works
            
    assert len(chunks) > 0, f"Streaming generation failed to yield any chunks for {engine_id}"
    assert chunks[0].dtype == np.float32 or chunks[0].dtype == np.uint8 # Standardized should be float32 but binary is sent
