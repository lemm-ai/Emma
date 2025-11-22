"""
Base model class for EMMA
Provides common interface for all AI models
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging
import torch

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all AI models in EMMA"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """
        Initialize base model
        
        Args:
            model_path: Path to model weights
            device: Device to run model on (cuda, rocm, cpu)
        """
        self.model_path = model_path
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None  # For models that need tokenizers
        self.processor = None  # For models that need processors
        self.is_loaded = False
        self.use_fallback = False  # Track if using fallback model
        
    def _get_device(self, device_str: str) -> torch.device:
        """Get torch device object"""
        if device_str in ["cuda", "rocm"]:
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                logger.warning(f"{device_str} requested but not available, using CPU")
                return torch.device("cpu")
        return torch.device("cpu")
    
    @abstractmethod
    def load(self) -> None:
        """Load model weights and prepare for inference"""
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Unload model from memory"""
        pass
    
    @abstractmethod
    def infer(self, *args, **kwargs) -> Any:
        """
        Run model inference
        
        Returns:
            Model output
        """
        pass
    
    def __enter__(self):
        """Context manager entry"""
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.unload()
