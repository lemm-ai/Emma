"""
Demucs wrapper for stem separation
Separates audio into vocals, drums, bass, and other stems
"""

import logging
from typing import Dict, List
import numpy as np
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class StemSeparator(BaseModel):
    """
    Wrapper for Demucs stem separation
    https://github.com/facebookresearch/demucs
    """
    
    def __init__(self, model_name: str = "htdemucs", device: str = "cuda"):
        super().__init__(model_path=None, device=device)
        self.model_name = model_name
        self.stem_names = ["vocals", "drums", "bass", "other"]
        
    def load(self) -> None:
        """Load Demucs model"""
        try:
            logger.info(f"Loading Demucs model: {self.model_name}...")
            
            import torch
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
            
            self.model = get_model(self.model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Store apply function for later use
            self.apply_fn = apply_model
            
            self.is_loaded = True
            logger.info("Demucs model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Demucs model: {e}")
            raise
    
    def unload(self) -> None:
        """Unload model from memory"""
        if self.model is not None:
            del self.model
            self.model = None
            self.is_loaded = False
            logger.info("Demucs model unloaded")
    
    def infer(self, audio: np.ndarray, sample_rate: int = 48000) -> Dict[str, np.ndarray]:
        """
        Separate audio into stems
        
        Args:
            audio: Input audio as numpy array (channels, samples)
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary mapping stem names to audio arrays
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            logger.info("Separating audio into stems...")
            
            import torch
            
            # Ensure audio is in correct format (channels, samples)
            if audio.ndim == 1:
                audio = audio[np.newaxis, :]
            elif audio.ndim == 2 and audio.shape[0] > audio.shape[1]:
                # Transpose if (samples, channels)
                audio = audio.T
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
            
            # Apply model
            with torch.no_grad():
                stems = self.apply_fn(self.model, audio_tensor, device=self.device)
            
            # Convert results to numpy
            result = {}
            stems = stems.squeeze(0).cpu().numpy()
            for i, stem_name in enumerate(self.stem_names):
                result[stem_name] = stems[i]
            
            logger.info("Stem separation complete")
            return result
            
        except Exception as e:
            logger.error(f"Error during stem separation: {e}")
            raise
    
    def recombine_stems(self, stems: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Recombine separated stems into single audio
        
        Args:
            stems: Dictionary of stem audio arrays
            
        Returns:
            Combined audio
        """
        combined = np.zeros_like(next(iter(stems.values())))
        
        for stem_audio in stems.values():
            combined += stem_audio
        
        return combined
