"""
LyricsMindAI wrapper
Generates lyrics from prompts
"""

import logging
from typing import Optional
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class LyricsGenerator(BaseModel):
    """
    Wrapper for LyricsMindAI
    https://github.com/AmirHaytham/LyricMind-AI
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        super().__init__(model_path, device)
        
    def load(self) -> None:
        """Load lyrics generation model"""
        try:
            import os
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Check if running on HuggingFace Spaces
            is_hf_space = os.getenv('SPACE_ID') is not None
            
            # Try to load LyricsMindAI first (local/preferred)
            if not is_hf_space and self.model_path:
                logger.info("Loading LyricsMindAI model...")
                try:
                    # TODO: Implement actual LyricsMindAI loading when available
                    # from lyricmind import LyricGenerator
                    # self.tokenizer = LyricGenerator.get_tokenizer()
                    # self.model = LyricGenerator.from_pretrained(self.model_path)
                    # self.model = self.model.to(self.device)
                    # self.model.eval()
                    # self.use_fallback = False
                    
                    # For now, raise to trigger fallback
                    raise ImportError("LyricsMindAI not yet available")
                    
                except (ImportError, FileNotFoundError) as e:
                    logger.warning(f"LyricsMindAI not available: {e}. Using fallback model.")
                    is_hf_space = True  # Force fallback
            
            # Fallback to GPT-2 on HuggingFace Spaces or when LyricsMindAI unavailable
            if is_hf_space or not self.model_path:
                logger.info("Loading GPT-2 fallback model for lyrics...")
                model_name = "gpt2"
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                self.model = self.model.to(self.device)
                self.model.eval()
                self.use_fallback = True
                
                # Set pad token if not set
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                logger.warning("Using GPT-2 as fallback. Note: Not optimized for song lyrics.")
            
            self.is_loaded = True
            logger.info("Lyrics generation model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading lyrics generation model: {e}")
            raise
    
    def unload(self) -> None:
        """Unload model from memory"""
        if self.model is not None:
            del self.model
            self.model = None
            self.is_loaded = False
            logger.info("LyricsMind AI model unloaded")
    
    def infer(
        self,
        prompt: str,
        max_length: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate lyrics from prompt
        
        Args:
            prompt: Text prompt describing desired lyrics
            max_length: Maximum length of generated lyrics
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Generated lyrics as string
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            logger.info(f"Generating lyrics from prompt: {prompt[:50]}...")
            
            import torch
            
            # Format prompt for lyrics generation
            formatted_prompt = f"Write lyrics for a song about {prompt}:\n\n"
            
            # Tokenize input
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            # Decode output
            lyrics = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from output
            if lyrics.startswith(formatted_prompt):
                lyrics = lyrics[len(formatted_prompt):]
            
            logger.info("Lyrics generation complete")
            return lyrics.strip()
            
        except Exception as e:
            logger.error(f"Error during lyrics generation: {e}")
            raise
