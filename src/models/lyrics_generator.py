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
        """Load LyricsMindAI model"""
        try:
            logger.info("Loading Lyrics generation model...")
            
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            # Use a pretrained model for lyrics generation
            # We'll use GPT-2 fine-tuned on lyrics or a similar model
            model_name = self.model_path or "gpt2"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
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
