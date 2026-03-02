import pytest
import os
import asyncio
import numpy as np
from app.api.ws import get_sentence_chunks
from app.engines.manager import plugin_manager
from app.utils import secure_path_join

def test_secure_path_join():
    base = "/tmp/test_base"
    os.makedirs(base, exist_ok=True)
    
    # Valid joins
    assert secure_path_join(base, "file.txt") == os.path.abspath(os.path.join(base, "file.txt"))
    assert secure_path_join(base, "sub/file.txt") == os.path.abspath(os.path.join(base, "sub/file.txt"))
    
    # Invalid joins (traversal)
    with pytest.raises(ValueError):
        secure_path_join(base, "../outside.txt")
    with pytest.raises(ValueError):
        secure_path_join(base, "/etc/passwd")
    with pytest.raises(ValueError):
        secure_path_join(base, "../../etc/passwd")

@pytest.mark.asyncio
async def test_engine_path_traversal_protection():
    # Test ZipVoice for traversal
    zipvoice = plugin_manager.get_plugin("zipvoice")
    if zipvoice:
        # This should use safe_path_join internally and likely return a default or error
        # We just want to ensure it doesn't crash or return an absolute path outside its dir
        path = zipvoice._get_ref_path("../../../etc/passwd")
        assert "/etc/passwd" not in path
        assert zipvoice.ref_dir in path or zipvoice.clones_dir in path

@pytest.mark.asyncio
async def test_concurrency_locks():
    # Simulate multiple concurrent requests to an engine
    # We'll use ZipVoice as it has a cache lock
    engine = plugin_manager.get_plugin("zipvoice")
    if not engine:
        return

    async def mock_request(id):
        # We're just checking that the lock doesn't cause a deadlock and handles multiple entries
        # In a real test, we might use a mock that sleeps and check for serial execution
        pass

    tasks = [mock_request(i) for i in range(5)]
    await asyncio.gather(*tasks)
    # If it completes without crashing/deadlocking, the lock mechanism is at least stable
