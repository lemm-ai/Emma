"""
Device detection and management utilities
Handles CUDA, ROCm, and CPU detection
"""

import torch
import logging
from typing import Literal, Optional

logger = logging.getLogger(__name__)

DeviceType = Literal["cuda", "rocm", "cpu"]


def detect_device() -> DeviceType:
    """
    Detect available compute device
    
    Returns:
        Device type: 'cuda', 'rocm', or 'cpu'
    """
    if torch.cuda.is_available():
        # Check if ROCm
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            logger.info("ROCm detected")
            return "rocm"
        else:
            logger.info("CUDA detected")
            return "cuda"
    else:
        logger.info("No GPU detected, using CPU")
        return "cpu"


def get_torch_device(device_type: Optional[str] = None) -> torch.device:
    """
    Get PyTorch device object
    
    Args:
        device_type: Optional device type override
        
    Returns:
        PyTorch device object
    """
    if device_type is None:
        device_type = detect_device()
    
    if device_type in ["cuda", "rocm"]:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            return device
        else:
            logger.warning("GPU requested but not available, falling back to CPU")
            return torch.device("cpu")
    else:
        return torch.device("cpu")


def get_device_info() -> dict:
    """
    Get detailed device information
    
    Returns:
        Dictionary with device information
    """
    info = {
        "device_type": detect_device(),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            info["rocm_version"] = torch.version.hip
    
    return info
