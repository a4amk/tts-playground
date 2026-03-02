import os
import torch
import logging

logger = logging.getLogger(__name__)

def secure_path_join(base_dir: str, filename: str) -> str:
    """
    Safely joins a base directory and a filename to prevent path traversal.
    Returns the absolute path if it is within base_dir, otherwise raises ValueError.
    """
    base_dir = os.path.abspath(base_dir)
    # Normalize the joined path
    joined_path = os.path.abspath(os.path.join(base_dir, filename))
    
    # Check if the joined path starts with the base_dir
    # commonpath returns the longest common sub-path
    if os.path.commonpath([base_dir, joined_path]) != base_dir:
        logger.warning(f"SECURITY: Blocked path traversal attempt! Base: {base_dir}, Attempt: {filename}")
        raise ValueError(f"Invalid path or filename: {filename}")
        
    return joined_path

def safe_torch_load(path: str, device: str = "cpu", weights_only: bool = True):
    """
    Wraps torch.load with weights_only=True by default for security.
    """
    try:
        return torch.load(path, map_location=device, weights_only=weights_only)
    except Exception as e:
        if weights_only:
            logger.warning(f"Standard weights_only=True failed for {path}. Retrying with weights_only=False carefully.")
            return torch.load(path, map_location=device, weights_only=False)
        raise e
