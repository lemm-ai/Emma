"""
Audio utility functions
Common audio processing helpers
"""

import numpy as np
from typing import Tuple, Optional
import soundfile as sf
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_audio(file_path: str, sample_rate: int = 48000) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample if needed
    
    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        audio, sr = sf.read(file_path)
        
        # Resample if needed
        if sr != sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            sr = sample_rate
            
        return audio, sr
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {e}")
        raise


def save_audio(
    audio: np.ndarray,
    file_path: str,
    sample_rate: int = 48000,
    format: str = "wav"
) -> None:
    """
    Save audio to file
    
    Args:
        audio: Audio data as numpy array
        file_path: Output file path
        sample_rate: Sample rate
        format: Output format (wav or mp3)
    """
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "mp3":
            # Convert to mp3 using pydub
            from pydub import AudioSegment
            
            # Save as temp wav first
            temp_wav = str(Path(file_path).with_suffix('.temp.wav'))
            sf.write(temp_wav, audio, sample_rate)
            
            # Convert to mp3
            sound = AudioSegment.from_wav(temp_wav)
            sound.export(file_path, format="mp3", bitrate="320k")
            
            # Clean up temp file
            Path(temp_wav).unlink()
        else:
            sf.write(file_path, audio, sample_rate)
            
        logger.info(f"Audio saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving audio to {file_path}: {e}")
        raise


def normalize_audio(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """
    Normalize audio to target dB level
    
    Args:
        audio: Audio data
        target_db: Target dB level
        
    Returns:
        Normalized audio
    """
    # Calculate current RMS
    rms = np.sqrt(np.mean(audio**2))
    
    if rms == 0:
        return audio
    
    # Calculate target RMS from dB
    target_rms = 10 ** (target_db / 20)
    
    # Apply gain
    gain = target_rms / rms
    normalized = audio * gain
    
    # Clip to prevent clipping
    normalized = np.clip(normalized, -1.0, 1.0)
    
    return normalized


def get_audio_duration(file_path: str) -> float:
    """
    Get audio file duration in seconds
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Duration in seconds
    """
    try:
        info = sf.info(file_path)
        return info.duration
    except Exception as e:
        logger.error(f"Error getting audio duration: {e}")
        return 0.0


def convert_to_stereo(audio: np.ndarray) -> np.ndarray:
    """
    Convert mono audio to stereo
    
    Args:
        audio: Audio data (mono or stereo)
        
    Returns:
        Stereo audio
    """
    if audio.ndim == 1:
        # Mono to stereo
        return np.stack([audio, audio], axis=1)
    elif audio.ndim == 2 and audio.shape[1] == 1:
        # Single channel to stereo
        return np.concatenate([audio, audio], axis=1)
    else:
        # Already stereo or multi-channel
        return audio
