"""
Configuration settings for EMMA
Loads from config.yaml and environment variables
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging
from ..utils.device_utils import detect_device

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration settings"""
    ace_step_model_path: str = ""
    lyrics_mind_model_path: str = ""
    demucs_model: str = "htdemucs"
    sovits_model_path: str = ""
    musiccontrolnet_model_path: str = ""
    audiosr_model_path: str = ""
    device: str = "auto"  # auto, cuda, rocm, or cpu
    use_gpu: bool = True
    
    # Fallback model settings for HuggingFace Spaces compatibility
    use_fallback_models: bool = False  # Auto-detected based on environment
    fallback_music_model: str = "facebook/musicgen-large"  # Fallback when ACE-Step unavailable
    fallback_lyrics_model: str = "gpt2"  # Fallback when LyricsMindAI unavailable
    

@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 48000
    clip_duration: int = 32
    lead_in_duration: int = 2
    lead_out_duration: int = 2
    core_duration: int = 28
    output_format: str = "wav"  # wav or mp3
    num_stems: int = 4  # vocals, drums, bass, other
    

@dataclass
class DatabaseConfig:
    """Database configuration"""
    db_type: str = "sqlite"  # sqlite or firebase
    sqlite_path: str = "data/emma_clips.db"
    firebase_config: Optional[Dict[str, Any]] = None
    

@dataclass
class UIConfig:
    """UI configuration"""
    theme: str = "default"
    share: bool = False
    server_name: str = "0.0.0.0"
    server_port: int = 7860
    

class Settings:
    """Main settings class"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config()
        self.config = self._load_config()
        
        # Initialize sub-configs
        self.model = ModelConfig(**self.config.get('model', {}))
        
        # Handle auto device detection
        if self.model.device == "auto":
            self.model.device = detect_device()
            logger.info(f"Auto-detected device: {self.model.device}")
            
        self.audio = AudioConfig(**self.config.get('audio', {}))
        self.database = DatabaseConfig(**self.config.get('database', {}))
        self.ui = UIConfig(**self.config.get('ui', {}))
        
        # Override with environment variables
        self._load_env_overrides()
        
    def _find_config(self) -> Path:
        """Find config.yaml in project root"""
        current = Path(__file__).parent
        while current != current.parent:
            config_file = current / "config.yaml"
            if config_file.exists():
                return config_file
            current = current.parent
        
        # Default location
        return Path(__file__).parent.parent.parent / "config.yaml"
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f) or {}
            else:
                logger.warning(f"Config file not found: {self.config_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def _load_env_overrides(self):
        """Load environment variable overrides"""
        # Model overrides
        device = os.getenv('EMMA_DEVICE')
        if device:
            self.model.device = device
        use_gpu = os.getenv('EMMA_USE_GPU')
        if use_gpu:
            self.model.use_gpu = use_gpu.lower() == 'true'
            
        # Database overrides
        db_type = os.getenv('EMMA_DB_TYPE')
        if db_type:
            self.database.db_type = db_type
        sqlite_path = os.getenv('EMMA_SQLITE_PATH')
        if sqlite_path:
            self.database.sqlite_path = sqlite_path
            
        # UI overrides
        server_port = os.getenv('EMMA_SERVER_PORT')
        if server_port:
            self.ui.server_port = int(server_port)
        share = os.getenv('EMMA_SHARE')
        if share:
            self.ui.share = share.lower() == 'true'


# Global settings instance
settings = Settings()
