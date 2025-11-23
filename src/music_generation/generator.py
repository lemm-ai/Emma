"""
Music generation pipeline
Orchestrates the complete music generation workflow
"""

import logging
from typing import Optional, Dict, Any
import numpy as np
from pathlib import Path

from ..models.ace_step_wrapper import ACEStepModel
from ..models.lyrics_generator import LyricsGenerator
from ..models.stem_separator import StemSeparator
from ..config.settings import settings

# Import spaces for GPU decoration
try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False

logger = logging.getLogger(__name__)


class MusicGenerator:
    """
    Main music generation pipeline
    Combines lyrics generation, music generation, and stem separation
    """
    
    def __init__(self):
        self.lyrics_gen = None
        self.music_gen = None
        self.stem_sep = None
        self.device = settings.model.device
        
    def initialize(self):
        """Initialize all models"""
        logger.info("Initializing music generation pipeline...")
        
        try:
            # Initialize lyrics generator
            self.lyrics_gen = LyricsGenerator(
                model_path=settings.model.lyrics_mind_model_path,
                device=self.device
            )
            self.lyrics_gen.load()
            
            # Initialize music generator
            # Support ONNX/DirectML via ONNXModel wrapper when device hints at onnx
            if 'onnx' in (self.device or '').lower() or 'directml' in (self.device or '').lower():
                try:
                    from ..models.onnx_wrapper import ONNXModel
                    logger.info("Using ONNX Runtime music model wrapper based on requested device")
                    self.music_gen = ONNXModel(
                        model_path=settings.model.ace_step_model_path,
                        device=self.device
                    )
                    self.music_gen.load()
                except Exception as e:
                    logger.warning(f"Failed to initialize ONNXModel for device {self.device}: {e}. Falling back to ACE-Step/MusicGen path.")
                    self.music_gen = ACEStepModel(
                        model_path=settings.model.ace_step_model_path,
                        device=self.device
                    )
                    self.music_gen.load()
            else:
                self.music_gen = ACEStepModel(
                    model_path=settings.model.ace_step_model_path,
                    device=self.device
                )
                self.music_gen.load()
            
            # Initialize stem separator
            self.stem_sep = StemSeparator(
                model_name=settings.model.demucs_model,
                device=self.device
            )
            self.stem_sep.load()
            
            logger.info("Music generation pipeline initialized")
        except Exception as e:
            logger.error(f"Error initializing pipeline: {e}")
            raise
    
    def cleanup(self):
        """Clean up and unload models"""
        if self.lyrics_gen:
            self.lyrics_gen.unload()
        if self.music_gen:
            self.music_gen.unload()
        if self.stem_sep:
            self.stem_sep.unload()
        logger.info("Music generation pipeline cleaned up")
    
    def generate_lyrics(self, prompt: str, **kwargs) -> str:
        """
        Generate lyrics from prompt
        
        Args:
            prompt: Text prompt for lyrics
            **kwargs: Additional parameters
            
        Returns:
            Generated lyrics
        """
        if not self.lyrics_gen or not self.lyrics_gen.is_loaded:
            raise RuntimeError("Lyrics generator not initialized")
        
        return self.lyrics_gen.infer(prompt, **kwargs)
    
    def generate_music(
        self,
        prompt: str,
        lyrics: Optional[str] = None,
        auto_generate_lyrics: bool = True,
        duration: int = 32,
        separate_stems: bool = True,
        reference_audio: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate complete music clip
        
        Args:
            prompt: Music generation prompt
            lyrics: Optional lyrics (auto-generated if not provided)
            auto_generate_lyrics: Whether to auto-generate lyrics
            duration: Clip duration in seconds
            separate_stems: Whether to separate into stems
            reference_audio: Optional reference audio for style consistency (channels, samples)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing audio, stems, and metadata
        """
        if not self.music_gen or not self.music_gen.is_loaded:
            raise RuntimeError("Music generator not initialized")
        
        try:
            # Generate lyrics if needed
            if lyrics is None and auto_generate_lyrics:
                logger.info("Auto-generating lyrics from prompt...")
                lyrics = self.generate_lyrics(prompt)
            
            # Generate music with optional reference audio for consistency
            logger.info("Generating music...")
            audio = self.music_gen.infer(
                prompt=prompt,
                lyrics=lyrics,
                duration=duration,
                reference_audio=reference_audio,
                **kwargs
            )
            
            # Get clip segments
            segments = self.music_gen.get_clip_segments(audio)
            
            # Separate stems if requested
            stems = None
            if separate_stems and self.stem_sep and self.stem_sep.is_loaded:
                logger.info("Separating audio into stems...")
                stems = self.stem_sep.infer(audio)
            
            # Calculate actual duration from audio shape
            # Audio shape is (channels, samples) from MusicGen
            if audio.ndim == 2:
                actual_duration = audio.shape[1] / settings.audio.sample_rate
            else:
                actual_duration = audio.shape[0] / settings.audio.sample_rate
            
            result = {
                'audio': audio,
                'segments': segments,
                'stems': stems,
                'lyrics': lyrics,
                'prompt': prompt,
                'duration': actual_duration,
                'sample_rate': settings.audio.sample_rate
            }
            
            logger.info(f"Music generation complete - duration: {actual_duration:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error during music generation: {e}")
            raise
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
