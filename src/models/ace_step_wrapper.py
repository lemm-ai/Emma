"""
ACE-Step model wrapper
Handles music generation with lyrics and prompts
"""

import logging
from typing import Optional, Dict, Any
import numpy as np
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class ACEStepModel(BaseModel):
    """
    Wrapper for ACE-Step music generation model
    https://github.com/ace-step/ACE-Step
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        super().__init__(model_path, device)
        self.clip_duration = 32  # Total clip duration in seconds
        self.lead_in = 2  # Lead-in duration
        self.lead_out = 2  # Lead-out duration
        self.core_duration = 28  # Core audio duration
        
    def load(self) -> None:
        """Load music generation model"""
        try:
            import os
            import torch
            
            # Check if running on HuggingFace Spaces
            is_hf_space = os.getenv('SPACE_ID') is not None
            
            # Try to load ACE-Step first (local/preferred)
            if not is_hf_space and self.model_path:
                logger.info("Loading ACE-Step model...")
                try:
                    # TODO: Implement actual ACE-Step loading when available
                    # from acestep import ACEStepGenerator
                    # self.processor = ACEStepGenerator.get_processor()
                    # self.model = ACEStepGenerator.from_pretrained(self.model_path)
                    # self.model = self.model.to(self.device)
                    # self.model.eval()
                    # self.use_fallback = False
                    
                    # For now, raise to trigger fallback
                    raise ImportError("ACE-Step not yet available")
                    
                except (ImportError, FileNotFoundError) as e:
                    logger.warning(f"ACE-Step not available: {e}. Using fallback model.")
                    is_hf_space = True  # Force fallback
            
            # Fallback to MusicGen on HuggingFace Spaces or when ACE-Step unavailable
            if is_hf_space or not self.model_path:
                logger.info("Loading MusicGen fallback model (large)...")
                from transformers import AutoProcessor, MusicgenForConditionalGeneration
                
                model_name = "facebook/musicgen-large"
                self.processor = AutoProcessor.from_pretrained(model_name)
                self.model = MusicgenForConditionalGeneration.from_pretrained(model_name)
                self.model = self.model.to(self.device)
                self.model.eval()
                self.use_fallback = True
                logger.warning("Using MusicGen-large as fallback. Note: Inferior quality and no vocals.")
            
            self.is_loaded = True
            logger.info("Music generation model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading music generation model: {e}")
            raise
    
    def unload(self) -> None:
        """Unload model from memory"""
        if self.model is not None:
            del self.model
            self.model = None
            self.is_loaded = False
            logger.info("ACE-Step model unloaded")
    
    def infer(
        self,
        prompt: str,
        lyrics: Optional[str] = None,
        duration: int = 32,
        **kwargs
    ) -> np.ndarray:
        """
        Generate music from prompt and lyrics
        
        Args:
            prompt: Text prompt for music generation
            lyrics: Optional lyrics text (will be combined with prompt)
            duration: Target duration in seconds
            **kwargs: Additional generation parameters
            
        Returns:
            Generated audio as numpy array (channels, samples)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            logger.info(f"Generating music with prompt: {prompt[:50]}...")
            
            import torch
            
            # Combine prompt with lyrics if provided
            full_prompt = prompt
            if lyrics:
                full_prompt = f"{prompt}. Lyrics: {lyrics[:200]}"  # Limit lyrics length
            
            # Process input
            inputs = self.processor(
                text=[full_prompt],
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            
            # Calculate number of tokens needed for duration
            # MusicGen generates at 50 Hz, so tokens = duration * 50
            max_new_tokens = int(duration * 50)
            
            # Generate music
            with torch.no_grad():
                audio_values = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    guidance_scale=3.0,
                    **kwargs
                )
            
            # Convert to numpy and reshape
            audio = audio_values[0].cpu().numpy()
            
            # Ensure stereo output
            if audio.ndim == 1:
                audio = np.stack([audio, audio])  # Mono to stereo
            elif audio.shape[0] == 1:
                audio = np.repeat(audio, 2, axis=0)
            
            logger.info("Music generation complete")
            return audio
            
        except Exception as e:
            logger.error(f"Error during music generation: {e}")
            raise
    
    def get_clip_segments(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Split generated clip into lead-in, core, and lead-out segments
        
        Args:
            audio: Full audio clip
            
        Returns:
            Dictionary with 'lead_in', 'core', 'lead_out' segments
        """
        sample_rate = 48000
        lead_in_samples = self.lead_in * sample_rate
        lead_out_samples = self.lead_out * sample_rate
        
        return {
            'lead_in': audio[:lead_in_samples],
            'core': audio[lead_in_samples:-lead_out_samples],
            'lead_out': audio[-lead_out_samples:],
            'full': audio
        }
