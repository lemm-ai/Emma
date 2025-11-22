"""
Timeline manager using PyDub
DAW-like timeline for arranging audio clips
"""

import logging
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass
from pydub import AudioSegment
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

ClipPosition = Literal["intro", "previous", "next", "outro"]


@dataclass
class TimelineClip:
    """Represents a clip on the timeline"""
    id: str
    audio_segment: AudioSegment
    start_time: float  # in seconds
    duration: float  # in seconds
    metadata: Dict[str, Any]


class TimelineManager:
    """
    Manages the song timeline
    Similar to video editor or DAW timeline
    """
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.clips: List[TimelineClip] = []
        self.total_duration: float = 0.0
        
    def add_clip(
        self,
        audio: np.ndarray,
        position: ClipPosition = "next",
        clip_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add audio clip to timeline
        
        Args:
            audio: Audio data as numpy array
            position: Where to place the clip (intro, previous, next, outro)
            clip_id: Optional clip ID
            metadata: Optional metadata for the clip
            
        Returns:
            Clip ID
        """
        import uuid
        
        if clip_id is None:
            clip_id = str(uuid.uuid4())
        
        if metadata is None:
            metadata = {}
        
        # Convert numpy array to AudioSegment
        audio_segment = self._numpy_to_audiosegment(audio)
        duration = len(audio_segment) / 1000.0  # Convert ms to seconds
        
        # Determine start time based on position
        if position == "intro":
            start_time = 0.0
            # Shift all existing clips
            for clip in self.clips:
                clip.start_time += duration
        elif position == "previous":
            if len(self.clips) > 0:
                # Insert before last clip
                last_clip = self.clips[-1]
                start_time = last_clip.start_time
                last_clip.start_time += duration
            else:
                start_time = 0.0
        elif position == "next":
            start_time = self.total_duration
        elif position == "outro":
            start_time = self.total_duration
        else:
            raise ValueError(f"Invalid position: {position}")
        
        clip = TimelineClip(
            id=clip_id,
            audio_segment=audio_segment,
            start_time=start_time,
            duration=duration,
            metadata=metadata
        )
        
        # Insert clip at appropriate position
        if position == "intro":
            self.clips.insert(0, clip)
        elif position == "previous" and len(self.clips) > 0:
            self.clips.insert(-1, clip)
        else:
            self.clips.append(clip)
        
        # Update total duration
        self._update_total_duration()
        
        logger.info(f"Added clip {clip_id} at position {position} (start: {start_time:.2f}s)")
        return clip_id
    
    def remove_clip(self, clip_id: str) -> bool:
        """
        Remove clip from timeline
        
        Args:
            clip_id: ID of clip to remove
            
        Returns:
            True if clip was removed, False if not found
        """
        for i, clip in enumerate(self.clips):
            if clip.id == clip_id:
                removed_clip = self.clips.pop(i)
                
                # Adjust positions of clips after this one
                for j in range(i, len(self.clips)):
                    self.clips[j].start_time -= removed_clip.duration
                
                self._update_total_duration()
                logger.info(f"Removed clip {clip_id}")
                return True
        
        logger.warning(f"Clip {clip_id} not found")
        return False
    
    def move_clip(self, clip_id: str, new_start_time: float) -> bool:
        """
        Move clip to new position on timeline
        
        Args:
            clip_id: ID of clip to move
            new_start_time: New start time in seconds
            
        Returns:
            True if successful
        """
        clip = self._get_clip(clip_id)
        if clip is None:
            return False
        
        clip.start_time = new_start_time
        # Re-sort clips by start time
        self.clips.sort(key=lambda c: c.start_time)
        self._update_total_duration()
        
        logger.info(f"Moved clip {clip_id} to {new_start_time:.2f}s")
        return True
    
    def clear(self):
        """Clear all clips from timeline"""
        self.clips.clear()
        self.total_duration = 0.0
        logger.info("Timeline cleared")
    
    def render(
        self,
        crossfade_duration: int = 2000,  # in milliseconds
        output_format: str = "wav"
    ) -> AudioSegment:
        """
        Render timeline to single audio file
        
        Args:
            crossfade_duration: Duration of crossfade between clips (ms)
            output_format: Output format (wav or mp3)
            
        Returns:
            Rendered AudioSegment
        """
        if len(self.clips) == 0:
            logger.warning("No clips to render")
            return AudioSegment.silent(duration=0)
        
        logger.info(f"Rendering timeline with {len(self.clips)} clips...")
        
        # Sort clips by start time
        sorted_clips = sorted(self.clips, key=lambda c: c.start_time)
        
        # Start with first clip
        result = sorted_clips[0].audio_segment
        
        # Add remaining clips with crossfade
        for i in range(1, len(sorted_clips)):
            clip = sorted_clips[i]
            
            # Calculate gap or overlap
            prev_end = sorted_clips[i-1].start_time + sorted_clips[i-1].duration
            gap = clip.start_time - prev_end
            
            if gap > 0:
                # Add silence
                result = result + AudioSegment.silent(duration=int(gap * 1000))
                result = result + clip.audio_segment
            else:
                # Crossfade
                result = result.append(clip.audio_segment, crossfade=crossfade_duration)
        
        logger.info(f"Timeline rendered: {len(result)/1000:.2f}s")
        return result
    
    def export(self, output_path: str, format: str = "wav") -> None:
        """
        Export timeline to audio file
        
        Args:
            output_path: Output file path
            format: Output format (wav or mp3)
        """
        rendered = self.render()
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "mp3":
            rendered.export(output_path, format="mp3", bitrate="320k")
        else:
            rendered.export(output_path, format="wav")
        
        logger.info(f"Timeline exported to {output_path}")
    
    def _get_clip(self, clip_id: str) -> Optional[TimelineClip]:
        """Get clip by ID"""
        for clip in self.clips:
            if clip.id == clip_id:
                return clip
        return None
    
    def _update_total_duration(self):
        """Update total timeline duration"""
        if len(self.clips) == 0:
            self.total_duration = 0.0
        else:
            last_clip = max(self.clips, key=lambda c: c.start_time + c.duration)
            self.total_duration = last_clip.start_time + last_clip.duration
    
    def _numpy_to_audiosegment(self, audio: np.ndarray) -> AudioSegment:
        """Convert numpy array to AudioSegment"""
        # Ensure audio is in correct format
        if audio.dtype != np.int16:
            # Convert float to int16
            if audio.dtype in [np.float32, np.float64]:
                audio = (audio * 32767).astype(np.int16)
            else:
                audio = audio.astype(np.int16)
        
        # Ensure stereo
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=1)
        
        # Create AudioSegment
        audio_segment = AudioSegment(
            audio.tobytes(),
            frame_rate=self.sample_rate,
            sample_width=audio.dtype.itemsize,
            channels=2
        )
        
        return audio_segment
    
    def get_info(self) -> Dict[str, Any]:
        """Get timeline information"""
        return {
            'total_duration': self.total_duration,
            'num_clips': len(self.clips),
            'clips': [
                {
                    'id': clip.id,
                    'start_time': clip.start_time,
                    'duration': clip.duration,
                    'metadata': clip.metadata
                }
                for clip in self.clips
            ]
        }
