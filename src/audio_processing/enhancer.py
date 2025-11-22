"""
Audio enhancement using Pedalboard
Provides DAW-like controls for audio processing
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
try:
    from pedalboard import (
        Pedalboard,
        Compressor,
        Gain,
        Reverb,
        Delay,
        Chorus,
        Phaser,
        Distortion,
        LadderFilter,
        Limiter,
        HighpassFilter,
        LowpassFilter,
        HighShelfFilter,
        LowShelfFilter,
        PeakFilter,
    )
except ImportError:
    from pedalboard._pedalboard import (
        Pedalboard,
        Compressor,
        Gain,
        Reverb,
        Delay,
        Chorus,
        Phaser,
        Distortion,
        LadderFilter,
        Limiter,
        HighpassFilter,
        LowpassFilter,
        HighShelfFilter,
        LowShelfFilter,
        PeakFilter,
    )

logger = logging.getLogger(__name__)


class AudioEnhancer:
    """
    Audio enhancement using Pedalboard
    https://github.com/spotify/pedalboard
    """
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.board = Pedalboard([])
        
    def apply_mastering_preset(
        self,
        audio: np.ndarray,
        preset: str = "balanced"
    ) -> np.ndarray:
        """
        Apply mastering preset to audio
        
        Args:
            audio: Input audio
            preset: Preset name (balanced, bright, warm, punchy, etc.)
            
        Returns:
            Processed audio
        """
        presets = {
            "balanced": self._mastering_balanced,
            "bright": self._mastering_bright,
            "warm": self._mastering_warm,
            "punchy": self._mastering_punchy,
            "soft": self._mastering_soft,
            "aggressive": self._mastering_aggressive,
            "vintage": self._mastering_vintage,
            "modern": self._mastering_modern,
            "radio": self._mastering_radio,
            "streaming": self._mastering_streaming,
            "bass_boost": self._mastering_bass_boost,
            "treble_boost": self._mastering_treble_boost,
            "wide_stereo": self._mastering_wide_stereo,
            "mono_compatible": self._mastering_mono_compatible,
            "loud": self._mastering_loud,
            "dynamic": self._mastering_dynamic,
        }
        
        if preset not in presets:
            logger.warning(f"Unknown preset: {preset}, using 'balanced'")
            preset = "balanced"
        
        logger.info(f"Applying mastering preset: {preset}")
        return presets[preset](audio)
    
    def _mastering_balanced(self, audio: np.ndarray) -> np.ndarray:
        """Balanced mastering preset"""
        board = Pedalboard([
            Compressor(threshold_db=-20, ratio=4, attack_ms=5, release_ms=100),
            Gain(gain_db=2),
            HighpassFilter(cutoff_frequency_hz=30),
            LowpassFilter(cutoff_frequency_hz=18000),
            Limiter(threshold_db=-1.0, release_ms=50),
        ])
        return board(audio, self.sample_rate)
    
    def _mastering_bright(self, audio: np.ndarray) -> np.ndarray:
        """Bright mastering preset"""
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=40),
            Gain(gain_db=1),
            Compressor(threshold_db=-18, ratio=3, attack_ms=3, release_ms=80),
            # Add treble boost (would need EQ plugin)
            Limiter(threshold_db=-0.5, release_ms=40),
        ])
        return board(audio, self.sample_rate)
    
    def _mastering_warm(self, audio: np.ndarray) -> np.ndarray:
        """Warm mastering preset"""
        board = Pedalboard([
            LowpassFilter(cutoff_frequency_hz=16000),
            Compressor(threshold_db=-22, ratio=3.5, attack_ms=8, release_ms=120),
            Gain(gain_db=1.5),
            Limiter(threshold_db=-1.5, release_ms=60),
        ])
        return board(audio, self.sample_rate)
    
    def _mastering_punchy(self, audio: np.ndarray) -> np.ndarray:
        """Punchy mastering preset"""
        board = Pedalboard([
            Compressor(threshold_db=-16, ratio=6, attack_ms=1, release_ms=60),
            Gain(gain_db=3),
            Limiter(threshold_db=-0.3, release_ms=30),
        ])
        return board(audio, self.sample_rate)
    
    def _mastering_soft(self, audio: np.ndarray) -> np.ndarray:
        """Soft mastering preset"""
        board = Pedalboard([
            Compressor(threshold_db=-24, ratio=2, attack_ms=10, release_ms=150),
            Gain(gain_db=1),
            Limiter(threshold_db=-2.0, release_ms=80),
        ])
        return board(audio, self.sample_rate)
    
    def _mastering_aggressive(self, audio: np.ndarray) -> np.ndarray:
        """Aggressive mastering preset"""
        board = Pedalboard([
            Compressor(threshold_db=-12, ratio=8, attack_ms=0.5, release_ms=40),
            Gain(gain_db=4),
            Limiter(threshold_db=-0.1, release_ms=20),
        ])
        return board(audio, self.sample_rate)
    
    def _mastering_vintage(self, audio: np.ndarray) -> np.ndarray:
        """Vintage mastering preset"""
        board = Pedalboard([
            LowpassFilter(cutoff_frequency_hz=14000),
            HighpassFilter(cutoff_frequency_hz=50),
            Compressor(threshold_db=-20, ratio=4, attack_ms=10, release_ms=100),
            Gain(gain_db=1),
            Limiter(threshold_db=-2.0, release_ms=70),
        ])
        return board(audio, self.sample_rate)
    
    def _mastering_modern(self, audio: np.ndarray) -> np.ndarray:
        """Modern mastering preset"""
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=25),
            Compressor(threshold_db=-18, ratio=4, attack_ms=3, release_ms=70),
            Gain(gain_db=2.5),
            Limiter(threshold_db=-0.3, release_ms=35),
        ])
        return board(audio, self.sample_rate)
    
    def _mastering_radio(self, audio: np.ndarray) -> np.ndarray:
        """Radio-ready mastering preset"""
        board = Pedalboard([
            Compressor(threshold_db=-14, ratio=6, attack_ms=2, release_ms=50),
            Gain(gain_db=3),
            Limiter(threshold_db=-0.1, release_ms=25),
        ])
        return board(audio, self.sample_rate)
    
    def _mastering_streaming(self, audio: np.ndarray) -> np.ndarray:
        """Streaming platform mastering preset"""
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=30),
            Compressor(threshold_db=-16, ratio=4, attack_ms=4, release_ms=80),
            Gain(gain_db=2),
            Limiter(threshold_db=-1.0, release_ms=45),
        ])
        return board(audio, self.sample_rate)
    
    def _mastering_bass_boost(self, audio: np.ndarray) -> np.ndarray:
        """Bass boost mastering preset"""
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=20),
            Compressor(threshold_db=-20, ratio=4, attack_ms=5, release_ms=100),
            Gain(gain_db=2),
            Limiter(threshold_db=-1.0, release_ms=50),
        ])
        return board(audio, self.sample_rate)
    
    def _mastering_treble_boost(self, audio: np.ndarray) -> np.ndarray:
        """Treble boost mastering preset"""
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=35),
            Compressor(threshold_db=-18, ratio=3.5, attack_ms=4, release_ms=85),
            Gain(gain_db=2),
            Limiter(threshold_db=-0.8, release_ms=40),
        ])
        return board(audio, self.sample_rate)
    
    def _mastering_wide_stereo(self, audio: np.ndarray) -> np.ndarray:
        """Wide stereo mastering preset"""
        board = Pedalboard([
            Chorus(rate_hz=1.0, depth=0.3, mix=0.3),
            Compressor(threshold_db=-20, ratio=4, attack_ms=5, release_ms=100),
            Gain(gain_db=2),
            Limiter(threshold_db=-1.0, release_ms=50),
        ])
        return board(audio, self.sample_rate)
    
    def _mastering_mono_compatible(self, audio: np.ndarray) -> np.ndarray:
        """Mono-compatible mastering preset"""
        board = Pedalboard([
            Compressor(threshold_db=-20, ratio=4, attack_ms=5, release_ms=100),
            Gain(gain_db=2),
            HighpassFilter(cutoff_frequency_hz=35),
            Limiter(threshold_db=-1.0, release_ms=50),
        ])
        return board(audio, self.sample_rate)
    
    def _mastering_loud(self, audio: np.ndarray) -> np.ndarray:
        """Loud mastering preset"""
        board = Pedalboard([
            Compressor(threshold_db=-14, ratio=8, attack_ms=1, release_ms=50),
            Gain(gain_db=4),
            Limiter(threshold_db=-0.1, release_ms=20),
        ])
        return board(audio, self.sample_rate)
    
    def _mastering_dynamic(self, audio: np.ndarray) -> np.ndarray:
        """Dynamic mastering preset"""
        board = Pedalboard([
            Compressor(threshold_db=-26, ratio=2.5, attack_ms=15, release_ms=150),
            Gain(gain_db=1),
            Limiter(threshold_db=-2.5, release_ms=80),
        ])
        return board(audio, self.sample_rate)
    
    def apply_custom_effects(
        self,
        audio: np.ndarray,
        effects_chain: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Apply custom effects chain
        
        Args:
            audio: Input audio
            effects_chain: List of effect configurations
            
        Returns:
            Processed audio
        """
        board = Pedalboard([])
        
        for effect_config in effects_chain:
            effect_type = effect_config.get('type')
            params = effect_config.get('params', {})
            
            # Add effect to board based on type
            if effect_type == 'compressor':
                board.append(Compressor(**params))
            elif effect_type == 'gain':
                board.append(Gain(**params))
            elif effect_type == 'reverb':
                board.append(Reverb(**params))
            elif effect_type == 'delay':
                board.append(Delay(**params))
            elif effect_type == 'limiter':
                board.append(Limiter(**params))
            # Add more effect types as needed
        
        return board(audio, self.sample_rate)
    
    def apply_custom_eq_and_effects(
        self,
        audio: np.ndarray,
        # EQ parameters
        low_shelf_gain: float = 0.0,  # dB, -12 to +12
        low_shelf_freq: float = 100.0,  # Hz
        mid1_gain: float = 0.0,  # dB, -12 to +12
        mid1_freq: float = 500.0,  # Hz
        mid1_q: float = 1.0,  # 0.1 to 10
        mid2_gain: float = 0.0,  # dB, -12 to +12
        mid2_freq: float = 2000.0,  # Hz
        mid2_q: float = 1.0,  # 0.1 to 10
        high_shelf_gain: float = 0.0,  # dB, -12 to +12
        high_shelf_freq: float = 8000.0,  # Hz
        # Effects parameters
        compressor_threshold: float = -20.0,  # dB
        compressor_ratio: float = 4.0,  # ratio
        compressor_attack: float = 5.0,  # ms
        compressor_release: float = 100.0,  # ms
        reverb_room_size: float = 0.0,  # 0.0 to 1.0
        reverb_damping: float = 0.5,  # 0.0 to 1.0
        reverb_wet_level: float = 0.0,  # 0.0 to 1.0
        delay_seconds: float = 0.0,  # seconds, 0.0 to 2.0
        delay_feedback: float = 0.0,  # 0.0 to 1.0
        delay_mix: float = 0.0,  # 0.0 to 1.0
        chorus_rate: float = 1.0,  # Hz
        chorus_depth: float = 0.0,  # 0.0 to 1.0
        chorus_mix: float = 0.0,  # 0.0 to 1.0
        distortion_drive: float = 1.0,  # 1.0 to 25.0
        gain_db: float = 0.0,  # dB, -20 to +20
        limiter_threshold: float = -1.0,  # dB
        limiter_release: float = 50.0,  # ms
    ) -> np.ndarray:
        """
        Apply custom EQ and effects chain to audio
        
        Args:
            audio: Input audio
            Various EQ and effects parameters (see parameter list above)
            
        Returns:
            Processed audio
        """
        board_effects = []
        
        # EQ Section (applied first)
        if low_shelf_gain != 0.0:
            board_effects.append(LowShelfFilter(cutoff_frequency_hz=low_shelf_freq, gain_db=low_shelf_gain, q=0.707))
        
        if mid1_gain != 0.0:
            board_effects.append(PeakFilter(cutoff_frequency_hz=mid1_freq, gain_db=mid1_gain, q=mid1_q))
        
        if mid2_gain != 0.0:
            board_effects.append(PeakFilter(cutoff_frequency_hz=mid2_freq, gain_db=mid2_gain, q=mid2_q))
        
        if high_shelf_gain != 0.0:
            board_effects.append(HighShelfFilter(cutoff_frequency_hz=high_shelf_freq, gain_db=high_shelf_gain, q=0.707))
        
        # Compressor (dynamics control)
        if compressor_threshold < 0.0:
            board_effects.append(Compressor(
                threshold_db=compressor_threshold,
                ratio=compressor_ratio,
                attack_ms=compressor_attack,
                release_ms=compressor_release
            ))
        
        # Distortion (saturation/overdrive)
        if distortion_drive > 1.0:
            board_effects.append(Distortion(drive_db=20.0 * np.log10(distortion_drive)))
        
        # Chorus (stereo widening/modulation)
        if chorus_mix > 0.0:
            board_effects.append(Chorus(rate_hz=chorus_rate, depth=chorus_depth, mix=chorus_mix))
        
        # Delay (echo)
        if delay_mix > 0.0 and delay_seconds > 0.0:
            board_effects.append(Delay(delay_seconds=delay_seconds, feedback=delay_feedback, mix=delay_mix))
        
        # Reverb (space/ambience)
        if reverb_wet_level > 0.0:
            board_effects.append(Reverb(
                room_size=reverb_room_size,
                damping=reverb_damping,
                wet_level=reverb_wet_level,
                dry_level=1.0 - reverb_wet_level
            ))
        
        # Gain (volume adjustment)
        if gain_db != 0.0:
            board_effects.append(Gain(gain_db=gain_db))
        
        # Limiter (final stage to prevent clipping)
        board_effects.append(Limiter(threshold_db=limiter_threshold, release_ms=limiter_release))
        
        # Create and apply the effects chain
        board = Pedalboard(board_effects)
        return board(audio, self.sample_rate)
