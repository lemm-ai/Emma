"""
Main Gradio UI for EMMA
Provides the user interface for music generation and editing
"""

import logging
import gradio as gr
from typing import Optional, Tuple, List
import numpy as np
from datetime import datetime
from pathlib import Path
import os

# Import spaces for HuggingFace ZeroGPU support
try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False
    # Create a no-op decorator for local development
    class spaces:
        @staticmethod
        def GPU(func):
            return func

from ..music_generation.generator import MusicGenerator
from ..audio_processing.enhancer import AudioEnhancer
from ..audio_processing.vocal_processor import VocalProcessor
from ..timeline.manager import TimelineManager
from ..clip_library.database import ClipLibrary, ClipMetadata
from ..config.settings import settings
from ..utils.logger import setup_logger

logger = logging.getLogger(__name__)


class EmmaUI:
    """Main UI class for EMMA"""
    
    def __init__(self):
        self.music_gen = MusicGenerator()
        self.audio_enhancer = AudioEnhancer()
        self.vocal_processor = VocalProcessor()
        self.timeline = TimelineManager()
        self.clip_library = ClipLibrary(settings.database.sqlite_path)
        
        # State
        self.current_clip = None
        self.current_stems = None
        
    def initialize(self):
        """Initialize all components"""
        logger.info("Initializing EMMA UI...")
        self.music_gen.initialize()
        logger.info("EMMA UI initialized")
    
    def cleanup(self):
        """Cleanup resources"""
        self.music_gen.cleanup()
    
    def generate_lyrics(self, prompt: str) -> str:
        """Generate lyrics from prompt"""
        try:
            if not prompt:
                return "Please enter a prompt first."
            
            lyrics = self.music_gen.generate_lyrics(prompt)
            return lyrics
        except Exception as e:
            logger.error(f"Error generating lyrics: {e}")
            return f"Error: {str(e)}"
    
    def generate_music(
        self,
        prompt: str,
        lyrics: str,
        timeline_position: str,
        auto_lyrics: bool,
        duration: int = 30
    ):
        """Generate music clip"""
        try:
            if not prompt:
                return None, "Please enter a prompt first."
            
            # Get reference audio from timeline for style consistency
            reference_audio = None
            timeline_data = self.timeline.get_info()
            if timeline_data and timeline_data.get('clips') and len(timeline_data['clips']) > 0:
                try:
                    # Export current timeline as temporary reference for MusicControlNet
                    logger.info("Exporting timeline as style reference for consistency...")
                    import tempfile
                    import os
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    temp_path = temp_file.name
                    temp_file.close()
                    
                    self.timeline.export(temp_path)
                    
                    # Load the exported audio
                    import soundfile as sf
                    timeline_audio, sr = sf.read(temp_path)
                    
                    # Clean up temp file
                    os.unlink(temp_path)
                    
                    # Convert to (channels, samples) format if needed
                    if timeline_audio.ndim == 1:
                        reference_audio = np.stack([timeline_audio, timeline_audio], axis=0)
                    elif timeline_audio.shape[1] == 2:  # (samples, channels)
                        reference_audio = timeline_audio.T
                    else:
                        reference_audio = timeline_audio
                    
                    logger.info(f"Using timeline as reference: shape {reference_audio.shape}")
                except Exception as e:
                    logger.warning(f"Could not export timeline as reference: {e}")
            
            # Generate music
            # Note: Disable stem separation on HF Spaces to avoid errors
            import os
            separate_stems = not os.getenv('SPACE_ID')
            
            result = self.music_gen.generate_music(
                prompt=prompt,
                lyrics=lyrics if lyrics else None,
                auto_generate_lyrics=auto_lyrics and not lyrics,
                duration=duration,
                separate_stems=separate_stems,
                reference_audio=reference_audio  # Pass for MusicControlNet/style consistency
            )
            
            logger.info(f"Generated result - audio shape: {result['audio'].shape}, duration: {result['duration']}")
            
            self.current_clip = result['audio']
            self.current_stems = result['stems']
            
            # Convert audio to correct format for Gradio (samples, channels)
            audio_for_gradio = result['audio']
            if audio_for_gradio.ndim == 2 and audio_for_gradio.shape[0] < audio_for_gradio.shape[1]:
                # If (channels, samples), transpose to (samples, channels)
                audio_for_gradio = audio_for_gradio.T
            elif audio_for_gradio.ndim == 1:
                # If mono, expand to stereo
                audio_for_gradio = np.stack([audio_for_gradio, audio_for_gradio], axis=1)
            
            logger.info(f"Audio for Gradio shape: {audio_for_gradio.shape}")
            print(f"[DEBUG] Audio for Gradio shape: {audio_for_gradio.shape}")
            
            # Add to timeline (use original audio in channels, samples format)
            from ..timeline.manager import ClipPosition
            position_map: dict[str, ClipPosition] = {
                "Intro": "intro",
                "Previous": "previous",
                "Next": "next",
                "Outro": "outro"
            }
            
            logger.info(f"Adding clip to timeline - audio shape: {result['audio'].shape}")
            print(f"[DEBUG] Adding clip to timeline - audio shape: {result['audio'].shape}")
            clip_id = self.timeline.add_clip(
                audio=result['audio'],
                position=position_map.get(timeline_position, "next"),
                metadata={
                    'prompt': prompt,
                    'lyrics': result['lyrics'],
                    'duration': result['duration']
                }
            )
            
            # Save to library
            output_path = f"output/clips/{clip_id}.wav"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            from ..utils.audio_utils import save_audio
            save_audio(result['audio'], output_path)
            
            metadata = ClipMetadata(
                clip_id=clip_id,
                name=f"Clip {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                prompt=prompt,
                lyrics=result['lyrics'],
                duration=result['duration'],
                bpm=None,
                key=None,
                genre=None,
                mood=None,
                created_at=datetime.now().isoformat(),
                file_path=output_path,
                tags=[],
                custom_data={}
            )
            
            self.clip_library.add_clip(metadata)
            
            # Get updated info displays
            clips_info = self.get_clip_library_info()
            timeline_html = self.get_timeline_html()
            library_html = self.get_clip_library_html()
            
            return (
                (settings.audio.sample_rate, audio_for_gradio),  # audio_output
                f"Music generated successfully! Clip ID: {clip_id}",  # status_text
                (settings.audio.sample_rate, audio_for_gradio),  # enhance_audio_input
                timeline_html,  # timeline_display
                library_html,  # library_display
                ""  # upload_status (clear it)
            )
            
        except Exception as e:
            logger.error(f"Error generating music: {e}")
            return (
                None, 
                f"Error: {str(e)}", 
                None, 
                self.get_timeline_html(), 
                self.get_clip_library_html(),
                ""
            )
    
    def apply_mastering(
        self,
        audio_input,
        preset: str
    ):
        """Apply mastering preset"""
        try:
            if audio_input is None:
                return None, "No audio to process. Generate music first."
            
            sample_rate, audio = audio_input
            
            # Convert to float32 for Pedalboard (only supports float32/float64)
            if audio.dtype not in [np.float32, np.float64]:
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
                elif audio.dtype == np.int32:
                    audio = audio.astype(np.float32) / 2147483648.0
                else:
                    audio = audio.astype(np.float32)
            
            processed = self.audio_enhancer.apply_mastering_preset(audio, preset.lower())
            
            return (sample_rate, processed), f"Applied {preset} mastering"
            
        except Exception as e:
            logger.error(f"Error applying mastering: {e}")
            return None, f"Error: {str(e)}"
    
    def apply_custom_effects(
        self,
        audio_input,
        low_shelf_gain, low_shelf_freq,
        mid1_gain, mid1_freq, mid1_q,
        mid2_gain, mid2_freq, mid2_q,
        high_shelf_gain, high_shelf_freq,
        comp_threshold, comp_ratio, comp_attack, comp_release,
        reverb_room, reverb_damping, reverb_wet,
        delay_time, delay_feedback, delay_mix,
        chorus_rate, chorus_depth, chorus_mix,
        distortion_drive, gain_db,
        limiter_threshold, limiter_release
    ):
        """Apply custom EQ and effects"""
        try:
            if audio_input is None:
                return None, "No audio to process. Generate music first."
            
            sample_rate, audio = audio_input
            
            # Convert to float32 for Pedalboard (only supports float32/float64)
            if audio.dtype not in [np.float32, np.float64]:
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
                elif audio.dtype == np.int32:
                    audio = audio.astype(np.float32) / 2147483648.0
                else:
                    audio = audio.astype(np.float32)
            
            processed = self.audio_enhancer.apply_custom_eq_and_effects(
                audio,
                low_shelf_gain=low_shelf_gain,
                low_shelf_freq=low_shelf_freq,
                mid1_gain=mid1_gain,
                mid1_freq=mid1_freq,
                mid1_q=mid1_q,
                mid2_gain=mid2_gain,
                mid2_freq=mid2_freq,
                mid2_q=mid2_q,
                high_shelf_gain=high_shelf_gain,
                high_shelf_freq=high_shelf_freq,
                compressor_threshold=comp_threshold,
                compressor_ratio=comp_ratio,
                compressor_attack=comp_attack,
                compressor_release=comp_release,
                reverb_room_size=reverb_room,
                reverb_damping=reverb_damping,
                reverb_wet_level=reverb_wet,
                delay_seconds=delay_time,
                delay_feedback=delay_feedback,
                delay_mix=delay_mix,
                chorus_rate=chorus_rate,
                chorus_depth=chorus_depth,
                chorus_mix=chorus_mix,
                distortion_drive=distortion_drive,
                gain_db=gain_db,
                limiter_threshold=limiter_threshold,
                limiter_release=limiter_release
            )
            
            return (sample_rate, processed), "Applied custom EQ and effects"
            
        except Exception as e:
            logger.error(f"Error applying custom effects: {e}")
            return None, f"Error: {str(e)}"
    
    def get_clip_library_info(self):
        """Get clip library information as DataFrame data"""
        try:
            clips = self.clip_library.search_clips("")  # Empty string returns all clips
            if not clips:
                return []
            
            # Return data as list of lists for DataFrame
            data = []
            for clip in clips:
                data.append([
                    clip.clip_id[:8] + "...",  # Shortened ID
                    clip.name,
                    clip.prompt[:50] + "..." if len(clip.prompt) > 50 else clip.prompt,
                    f"{clip.duration:.2f}s",
                    clip.created_at.split("T")[0] if "T" in clip.created_at else clip.created_at[:10]
                ])
            
            return data
        except Exception as e:
            logger.error(f"Error getting clip library info: {e}")
            return []
    
    def get_timeline_html(self):
        """Generate HTML for visual timeline with bubble-style clips"""
        try:
            timeline_data = self.timeline.get_info()
            if not timeline_data or not timeline_data.get('clips'):
                return """
                <div style="padding: 40px; text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; color: white;">
                    <h3>🎵 Timeline Empty</h3>
                    <p>Generate music and it will appear here automatically!</p>
                </div>
                """
            
            total_duration = timeline_data.get('total_duration', 0)
            clips = timeline_data.get('clips', [])
            
            # Generate timeline ruler with better contrast
            ruler_html = '<div style="display: flex; margin-bottom: 8px; padding: 0 10px; font-size: 11px; color: #333; font-weight: 500; background: #e8e8e8; border-radius: 4px; padding: 4px 10px;">'
            num_markers = min(int(total_duration) + 1, 20)
            for i in range(num_markers):
                ruler_html += f'<div style="flex: 1; text-align: center;">{i}s</div>'
            ruler_html += '</div>'
            
            # Playback controls
            controls_html = '''
            <div style="display: flex; gap: 8px; margin-bottom: 10px; padding: 8px; background: #f0f0f0; border-radius: 8px;">
                <button onclick="playTimeline()" style="flex: 1; padding: 8px 16px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        border: none; border-radius: 6px; color: white; font-weight: bold; cursor: pointer; transition: transform 0.2s;"
                        onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                    ▶ Play
                </button>
                <button onclick="pauseTimeline()" style="flex: 1; padding: 8px 16px; background: #6c757d; 
                        border: none; border-radius: 6px; color: white; font-weight: bold; cursor: pointer; transition: transform 0.2s;"
                        onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                    ⏸ Pause
                </button>
                <button onclick="stopTimeline()" style="flex: 1; padding: 8px 16px; background: #dc3545; 
                        border: none; border-radius: 6px; color: white; font-weight: bold; cursor: pointer; transition: transform 0.2s;"
                        onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                    ⏹ Stop
                </button>
            </div>
            <script>
                function playTimeline() { alert('Timeline playback not yet implemented - export and use audio player'); }
                function pauseTimeline() { alert('Timeline playback not yet implemented - export and use audio player'); }
                function stopTimeline() { alert('Timeline playback not yet implemented - export and use audio player'); }
            </script>
            '''
            
            # Generate clip bubbles
            clips_html = '<div style="position: relative; height: 80px; background: #f5f5f5; border-radius: 8px; padding: 10px; margin-top: 5px;">'
            
            for clip in clips:
                clip_id = clip.get('id', 'Unknown')[:8]
                start_time = clip.get('start_time', 0)
                duration = clip.get('duration', 0)
                
                # Calculate position and width as percentage
                left_percent = (start_time / total_duration * 100) if total_duration > 0 else 0
                width_percent = (duration / total_duration * 100) if total_duration > 0 else 10
                
                # Generate random color based on clip_id
                hue = hash(clip.get('id', '')) % 360
                color = f"hsl({hue}, 70%, 60%)"
                
                clips_html += f'''
                <div style="position: absolute; left: {left_percent}%; width: {width_percent}%; height: 60px; 
                     background: {color}; border-radius: 12px; padding: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                     overflow: hidden; cursor: pointer; transition: transform 0.2s;" 
                     onmouseover="this.style.transform='scale(1.05)'" 
                     onmouseout="this.style.transform='scale(1)'">
                    <div style="font-weight: bold; font-size: 12px; color: white; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">
                        {clip_id}...
                    </div>
                    <div style="font-size: 10px; color: rgba(255,255,255,0.9);">
                        {duration:.1f}s
                    </div>
                </div>
                '''
            
            clips_html += '</div>'
            
            # Combine ruler, controls, and clips
            timeline_html = f'''
            <div style="background: white; border-radius: 12px; padding: 15px; box-shadow: 0 2px 12px rgba(0,0,0,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <h4 style="margin: 0; color: #333;">🎼 Timeline</h4>
                    <span style="color: #666; font-size: 13px;">Total: {total_duration:.1f}s | {len(clips)} clips</span>
                </div>
                {controls_html}
                {ruler_html}
                {clips_html}
            </div>
            '''
            
            return timeline_html
            
        except Exception as e:
            logger.error(f"Error generating timeline HTML: {e}")
            return f'<div style="padding: 20px; color: red;">Error loading timeline: {str(e)}</div>'
    
    def get_clip_library_html(self):
        """Generate HTML for clip library sidebar with bubble-style clips"""
        try:
            clips = self.clip_library.search_clips("")
            if not clips:
                return """
                <div style="padding: 20px; text-align: center; color: #888;">
                    <p>📚 No clips yet</p>
                    <p style="font-size: 12px;">Generate or upload clips to see them here</p>
                </div>
                """
            
            clips_html = '<div style="display: flex; flex-direction: column; gap: 12px; padding: 10px;">'
            
            for clip in clips:
                clip_id = clip.clip_id
                short_id = clip_id[:8]
                name = clip.name or "Untitled"
                duration = clip.duration
                prompt_preview = (clip.prompt[:40] + "...") if len(clip.prompt) > 40 else clip.prompt
                
                # Color based on clip_id
                hue = hash(clip_id) % 360
                color = f"hsl({hue}, 65%, 55%)"
                
                clips_html += f'''
                <div style="background: {color}; border-radius: 12px; padding: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); 
                     transition: transform 0.2s, box-shadow 0.2s;" 
                     onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.25)'"
                     onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 2px 8px rgba(0,0,0,0.15)'">
                    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 8px;">
                        <div style="flex: 1;">
                            <div style="font-weight: bold; color: white; font-size: 13px; text-shadow: 0 1px 2px rgba(0,0,0,0.2);">
                                {name}
                            </div>
                            <div style="font-size: 11px; color: rgba(255,255,255,0.85); margin-top: 2px;">
                                {short_id}... • {duration:.1f}s
                            </div>
                        </div>
                    </div>
                    <div style="font-size: 11px; color: rgba(255,255,255,0.9); margin-bottom: 8px; line-height: 1.3;">
                        {prompt_preview}
                    </div>
                    
                    <!-- Mini playback controls -->
                    <div style="display: flex; gap: 4px; margin-bottom: 8px; padding: 6px; background: rgba(0,0,0,0.15); border-radius: 6px;">
                        <button onclick="playClip('{clip_id}')" 
                                style="flex: 1; padding: 4px; background: rgba(255,255,255,0.2); border: none; border-radius: 4px; 
                                color: white; font-size: 16px; cursor: pointer; transition: background 0.2s;"
                                onmouseover="this.style.background='rgba(255,255,255,0.3)'"
                                onmouseout="this.style.background='rgba(255,255,255,0.2)'">
                            ▶
                        </button>
                        <button onclick="pauseClip('{clip_id}')" 
                                style="flex: 1; padding: 4px; background: rgba(255,255,255,0.2); border: none; border-radius: 4px; 
                                color: white; font-size: 16px; cursor: pointer; transition: background 0.2s;"
                                onmouseover="this.style.background='rgba(255,255,255,0.3)'"
                                onmouseout="this.style.background='rgba(255,255,255,0.2)'">
                            ⏸
                        </button>
                        <button onclick="stopClip('{clip_id}')" 
                                style="flex: 1; padding: 4px; background: rgba(255,255,255,0.2); border: none; border-radius: 4px; 
                                color: white; font-size: 16px; cursor: pointer; transition: background 0.2s;"
                                onmouseover="this.style.background='rgba(255,255,255,0.3)'"
                                onmouseout="this.style.background='rgba(255,255,255,0.2)'">
                            ⏹
                        </button>
                    </div>
                    
                    <!-- Icon action buttons -->
                    <div style="display: flex; gap: 6px; flex-wrap: wrap;">
                        <button onclick="handleClipAction('{clip_id}', 'rename')" title="Rename"
                                style="flex: 1; min-width: 40px; padding: 8px; background: rgba(255,255,255,0.25); 
                                border: none; border-radius: 6px; color: white; font-size: 16px; cursor: pointer; 
                                transition: background 0.2s;"
                                onmouseover="this.style.background='rgba(255,255,255,0.35)'"
                                onmouseout="this.style.background='rgba(255,255,255,0.25)'">
                            ✏️
                        </button>
                        <button onclick="handleClipAction('{clip_id}', 'extend')" title="Extend Clip"
                                style="flex: 1; min-width: 40px; padding: 8px; background: rgba(255,255,255,0.25); 
                                border: none; border-radius: 6px; color: white; font-size: 16px; cursor: pointer; 
                                transition: background 0.2s;"
                                onmouseover="this.style.background='rgba(255,255,255,0.35)'"
                                onmouseout="this.style.background='rgba(255,255,255,0.25)'">
                            🔄
                        </button>
                        <button onclick="handleClipAction('{clip_id}', 'download')" title="Download"
                                style="flex: 1; min-width: 40px; padding: 8px; background: rgba(255,255,255,0.25); 
                                border: none; border-radius: 6px; color: white; font-size: 16px; cursor: pointer;
                                transition: background 0.2s;"
                                onmouseover="this.style.background='rgba(255,255,255,0.35)'"
                                onmouseout="this.style.background='rgba(255,255,255,0.25)'">
                            📥
                        </button>
                        <button onclick="handleClipAction('{clip_id}', 'delete')" title="Delete"
                                style="flex: 1; min-width: 40px; padding: 8px; background: rgba(255,0,0,0.3); 
                                border: none; border-radius: 6px; color: white; font-size: 16px; cursor: pointer;
                                transition: background 0.2s;"
                                onmouseover="this.style.background='rgba(255,0,0,0.45)'"
                                onmouseout="this.style.background='rgba(255,0,0,0.3)'">
                            🗑️
                        </button>
                    </div>
                </div>
                <script>
                    function playClip(id) {{ alert('Clip playback not yet implemented. Use Download to get audio file.'); }}
                    function pauseClip(id) {{ alert('Clip playback not yet implemented'); }}
                    function stopClip(id) {{ alert('Clip playback not yet implemented'); }}
                    function handleClipAction(id, action) {{ 
                        if (action === 'extend') {{
                            alert('Extend: Clear timeline and place clip at start for extension (not yet implemented)');
                        }} else {{
                            alert(action.charAt(0).toUpperCase() + action.slice(1) + ' action not yet implemented'); 
                        }}
                    }}
                </script>
                '''
            
            clips_html += '</div>'
            
            header_html = '''
            <div style="padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                 border-radius: 12px 12px 0 0; color: white;">
                <h3 style="margin: 0; font-size: 16px;">📚 Clip Library</h3>
                <p style="margin: 5px 0 0 0; font-size: 12px; opacity: 0.9;">''' + f"{len(clips)} clips</p></div>"
            
            return header_html + clips_html
            
        except Exception as e:
            logger.error(f"Error generating clip library HTML: {e}")
            return f'<div style="padding: 20px; color: red;">Error: {str(e)}</div>'
        """Get formatted timeline information"""
        try:
            timeline_data = self.timeline.get_info()
            if not timeline_data or not timeline_data.get('clips'):
                return "Timeline is empty. Generate music and it will be added automatically."
            
            info_lines = [f"Total Duration: {timeline_data.get('total_duration', 0):.2f}s"]
            info_lines.append(f"Number of Clips: {timeline_data.get('num_clips', 0)}")
            
            return "\n".join(info_lines)
        except Exception as e:
            logger.error(f"Error getting timeline info: {e}")
            return f"Error loading timeline: {str(e)}"
    
    def get_timeline_clips_dataframe(self):
        """Get timeline clips as DataFrame data"""
        try:
            timeline_data = self.timeline.get_info()
            if not timeline_data or not timeline_data.get('clips'):
                return []
            
            # Return data as list of lists for DataFrame
            data = []
            for i, clip in enumerate(timeline_data['clips']):
                clip_name = clip.get('id', 'Unknown')[:8] + "..."
                start_time = clip.get('start_time', 0)
                duration = clip.get('duration', 0)
                end_time = start_time + duration
                data.append([
                    f"Track {i+1}",  # Track number
                    clip_name,
                    f"{start_time:.2f}",
                    f"{duration:.2f}",
                    f"{end_time:.2f}"
                ])
            
            return data
        except Exception as e:
            logger.error(f"Error getting timeline clips dataframe: {e}")
            return []
    
    def upload_to_timeline(self, audio_file):
        """Upload audio file and add to timeline"""
        try:
            if audio_file is None:
                return self.get_timeline_html(), self.get_clip_library_html(), "No file uploaded"
            
            # Load audio file
            import soundfile as sf
            audio, sample_rate = sf.read(audio_file)
            
            # Resample if needed
            from ..config.settings import Settings
            settings = Settings()
            if sample_rate != settings.audio.sample_rate:
                from scipy import signal
                num_samples = int(len(audio) * settings.audio.sample_rate / sample_rate)
                audio = signal.resample(audio, num_samples)
                sample_rate = settings.audio.sample_rate
            
            # Ensure correct format (channels, samples)
            if audio.ndim == 1:
                # Mono to stereo
                audio = np.stack([audio, audio], axis=0)
            elif audio.ndim == 2:
                # Check if it's (samples, channels) and transpose to (channels, samples)
                if audio.shape[1] == 2 and audio.shape[0] > audio.shape[1]:
                    audio = audio.T
            
            # Add to timeline at the end
            import os
            filename = os.path.basename(audio_file)
            clip_id = self.timeline.add_clip(
                audio=audio,
                position="outro",  # Add at the end
                metadata={'prompt': f"Uploaded: {filename}"}
            )
            
            # Also add to clip library
            output_path = f"output/clips/upload_{clip_id}.wav"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, audio.T if audio.shape[0] == 2 else audio, sample_rate)
            
            from ..clip_library.database import ClipMetadata
            metadata = ClipMetadata(
                clip_id=clip_id,
                name=filename,
                prompt=f"Uploaded file: {filename}",
                lyrics="",
                duration=audio.shape[1] / sample_rate if audio.ndim == 2 else len(audio) / sample_rate,
                bpm=None,
                key=None,
                genre=None,
                mood=None,
                created_at=datetime.now().isoformat(),
                file_path=output_path,
                tags=["uploaded"],
                custom_data={}
            )
            
            self.clip_library.add_clip(metadata)
            
            return (
                self.get_timeline_html(),
                self.get_clip_library_html(),
                f"Successfully added '{filename}' to timeline and library!"
            )
            
        except Exception as e:
            logger.error(f"Error uploading to timeline: {e}")
            return (
                self.get_timeline_html(),
                self.get_clip_library_html(),
                f"Error: {str(e)}"
            )
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface"""
        
        # Check if using fallback models
        import os
        is_hf_space = os.getenv('SPACE_ID') is not None
        fallback_warning = ""
        if is_hf_space:
            fallback_warning = """
            > ⚠️ **Note**: Running on HuggingFace Spaces with fallback models:
            > - Using MusicGen-Large (lower quality than ACE-Step, no vocal generation)
            > - Using GPT-2 for lyrics (not optimized for song lyrics like LyricsMindAI)
            > 
            > For best results, run EMMA locally with proper model weights.
            """
        
        with gr.Blocks(title="EMMA - Experimental Music Making Algorithm", css="""
            .clip-library-sidebar {
                position: fixed !important;
                right: 0;
                top: 0;
                height: 100vh;
                width: 320px;
                overflow-y: auto;
                background: #fafafa;
                border-left: 1px solid #e0e0e0;
                z-index: 1000;
            }
            .main-content {
                margin-right: 340px;
            }
        """) as app:
            # Main layout with sidebar
            with gr.Row():
                # Main content area
                with gr.Column(scale=4, elem_classes="main-content"):
                    gr.Markdown(f"""
                    # 🎵 EMMA - Experimental Music Making Algorithm
                    
                    AI-powered music generation and enhancement platform
                    
                    **Created by Gamahea / LEMM Project** | Making AI music free and open source for all
                    {fallback_warning}
                    """)
                    
                    # Visual Timeline (persistent across tabs)
                    timeline_display = gr.HTML(value=self.get_timeline_html(), label="Timeline")
                    
                    with gr.Tab("🎼 Generate Music"):
                        prompt_input = gr.Textbox(
                            label="Music Prompt",
                            placeholder="Describe the music you want to create (e.g., 'upbeat pop song with catchy melody')",
                            lines=3
                        )
                        
                        with gr.Row():
                            auto_lyrics_check = gr.Checkbox(
                                label="Auto-generate Lyrics",
                                value=False  # Disabled by default - MusicGen can't do vocals
                            )
                            gen_lyrics_btn = gr.Button("Generate Lyrics Only", size="sm")
                        
                        lyrics_input = gr.Textbox(
                            label="Lyrics (optional)",
                            placeholder="Enter your lyrics or let AI generate them...",
                            lines=8
                        )
                        
                        with gr.Row():
                            timeline_position = gr.Radio(
                                choices=["Intro", "Previous", "Next", "Outro"],
                                value="Next",
                                label="Timeline Position"
                            )
                            duration_slider = gr.Slider(
                                minimum=5,
                                maximum=300,
                                value=30,
                                step=5,
                                label="Duration (seconds)",
                                info="Generate clips from 5 seconds to 5 minutes"
                            )
                        
                        with gr.Row():
                            generate_btn = gr.Button("🎵 Generate Music", variant="primary", size="lg", scale=3)
                            upload_audio = gr.Audio(label="Or Upload Audio", type="filepath", scale=2)
                        
                        with gr.Row():
                            audio_output = gr.Audio(label="Generated Music")
                            status_text = gr.Textbox(label="Status", interactive=False)
                            error_log_textbox = gr.Textbox(label="Error Log", interactive=False, visible=False, lines=8)
                    
                    with gr.Tab("🎚️ Audio Enhancement"):
                        enhance_audio_input = gr.Audio(label="Audio Input")
                        
                        gr.Markdown("### Quick Mastering Presets")
                        mastering_preset = gr.Dropdown(
                            choices=[
                                "Balanced", "Bright", "Warm", "Punchy", "Soft",
                                "Aggressive", "Vintage", "Modern", "Radio", "Streaming",
                                "Bass Boost", "Treble Boost", "Wide Stereo",
                                "Mono Compatible", "Loud", "Dynamic"
                            ],
                            value="Balanced",
                            label="Mastering Preset"
                        )
                        
                        apply_mastering_btn = gr.Button("Apply Mastering", variant="primary")
                        
                        gr.Markdown("### Custom EQ & Effects")
                        
                        with gr.Accordion("🎛️ Equalizer", open=False):
                            with gr.Row():
                                low_shelf_gain = gr.Slider(-12, 12, 0, step=0.5, label="Low Shelf Gain (dB)")
                                low_shelf_freq = gr.Slider(20, 500, 100, step=10, label="Low Shelf Freq (Hz)")
                            with gr.Row():
                                mid1_gain = gr.Slider(-12, 12, 0, step=0.5, label="Mid 1 Gain (dB)")
                                mid1_freq = gr.Slider(200, 2000, 500, step=50, label="Mid 1 Freq (Hz)")
                                mid1_q = gr.Slider(0.1, 10, 1.0, step=0.1, label="Mid 1 Q")
                            with gr.Row():
                                mid2_gain = gr.Slider(-12, 12, 0, step=0.5, label="Mid 2 Gain (dB)")
                                mid2_freq = gr.Slider(1000, 8000, 2000, step=100, label="Mid 2 Freq (Hz)")
                                mid2_q = gr.Slider(0.1, 10, 1.0, step=0.1, label="Mid 2 Q")
                            with gr.Row():
                                high_shelf_gain = gr.Slider(-12, 12, 0, step=0.5, label="High Shelf Gain (dB)")
                                high_shelf_freq = gr.Slider(2000, 16000, 8000, step=500, label="High Shelf Freq (Hz)")
                        
                        with gr.Accordion("🎚️ Dynamics", open=False):
                            with gr.Row():
                                comp_threshold = gr.Slider(-60, 0, -20, step=1, label="Compressor Threshold (dB)")
                                comp_ratio = gr.Slider(1, 20, 4, step=0.5, label="Compressor Ratio")
                            with gr.Row():
                                comp_attack = gr.Slider(0.1, 100, 5, step=0.5, label="Attack (ms)")
                                comp_release = gr.Slider(10, 1000, 100, step=10, label="Release (ms)")
                        
                        with gr.Accordion("🌊 Reverb", open=False):
                            reverb_room = gr.Slider(0, 1, 0, step=0.01, label="Room Size")
                            reverb_damping = gr.Slider(0, 1, 0.5, step=0.01, label="Damping")
                            reverb_wet = gr.Slider(0, 1, 0, step=0.01, label="Wet Level")
                        
                        with gr.Accordion("🔁 Delay", open=False):
                            delay_time = gr.Slider(0, 2, 0, step=0.01, label="Delay Time (s)")
                            delay_feedback = gr.Slider(0, 0.95, 0, step=0.05, label="Feedback")
                            delay_mix = gr.Slider(0, 1, 0, step=0.01, label="Mix")
                        
                        with gr.Accordion("🎭 Modulation", open=False):
                            chorus_rate = gr.Slider(0.1, 10, 1, step=0.1, label="Chorus Rate (Hz)")
                            chorus_depth = gr.Slider(0, 1, 0, step=0.01, label="Chorus Depth")
                            chorus_mix = gr.Slider(0, 1, 0, step=0.01, label="Chorus Mix")
                        
                        with gr.Accordion("🔥 Distortion & Output", open=False):
                            distortion_drive = gr.Slider(1, 25, 1, step=0.5, label="Distortion Drive")
                            gain_db = gr.Slider(-20, 20, 0, step=0.5, label="Output Gain (dB)")
                            limiter_threshold = gr.Slider(-20, 0, -1, step=0.5, label="Limiter Threshold (dB)")
                            limiter_release = gr.Slider(10, 500, 50, step=10, label="Limiter Release (ms)")
                        
                        apply_custom_fx_btn = gr.Button("Apply Custom EQ & Effects", variant="secondary")
                        
                        enhanced_output = gr.Audio(label="Enhanced Audio")
                        enhance_status = gr.Textbox(label="Status", interactive=False)
                    
                    with gr.Tab("ℹ️ About"):
                        gr.Markdown("""
                        ## About EMMA
                        
                        EMMA (Experimental Music Making Algorithm) is an AI-powered music generation and enhancement platform.
                        
                        ### Features:
                        - **AI Lyrics Generation** - Generate lyrics from text prompts
                        - **Music Generation** - Create clips with AI
                        - **Stem Separation** - Separate vocals, drums, bass, and other instruments
                        - **Audio Enhancement** - Professional mastering presets + custom EQ/effects
                        - **Visual Timeline** - Arrange clips like a DAW
                        - **Clip Library** - Organize and manage your creations (see sidebar →)
                        
                        ### Technology Stack:
                        - ACE-Step / MusicGen for music generation
                        - LyricsMindAI / GPT-2 for lyrics generation
                        - Demucs for stem separation
                        - Pedalboard for audio processing
                        - Gradio for the user interface
                        
                        ### License
                        Apache 2.0 (or GPL 3.0 where required by dependencies)
                        
                        **Open Source | Free for All | Created by Gamahea / LEMM Project**
                        """)
                
                # Clip Library Sidebar (persistent across all tabs)
                with gr.Column(scale=1, elem_classes="clip-library-sidebar"):
                    library_display = gr.HTML(value=self.get_clip_library_html(), label="Clip Library")
                    upload_status = gr.Textbox(label="Upload Status", interactive=False, lines=2)
            
            # Connect button callbacks after all components are defined
            gen_lyrics_btn.click(
                fn=self.generate_lyrics,
                inputs=[prompt_input],
                outputs=[lyrics_input]
            )
            
            # Update generate button to refresh timeline and library displays
            generate_btn.click(
                fn=gpu_generate_music,
                inputs=[prompt_input, lyrics_input, timeline_position, auto_lyrics_check, duration_slider],
                outputs=[audio_output, status_text, enhance_audio_input, timeline_display, library_display, upload_status, error_log_textbox]
            )
            
            # Upload audio updates timeline and library
            upload_audio.change(
                fn=self.upload_to_timeline,
                inputs=[upload_audio],
                outputs=[timeline_display, library_display, upload_status]
            )
            
            apply_mastering_btn.click(
                fn=gpu_apply_mastering,
                inputs=[enhance_audio_input, mastering_preset],
                outputs=[enhanced_output, enhance_status]
            )
            
            apply_custom_fx_btn.click(
                fn=self.apply_custom_effects,
                inputs=[
                    enhance_audio_input,
                    low_shelf_gain, low_shelf_freq,
                    mid1_gain, mid1_freq, mid1_q,
                    mid2_gain, mid2_freq, mid2_q,
                    high_shelf_gain, high_shelf_freq,
                    comp_threshold, comp_ratio, comp_attack, comp_release,
                    reverb_room, reverb_damping, reverb_wet,
                    delay_time, delay_feedback, delay_mix,
                    chorus_rate, chorus_depth, chorus_mix,
                    distortion_drive, gain_db,
                    limiter_threshold, limiter_release
                ],
                outputs=[enhanced_output, enhance_status]
            )
            
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                ## About EMMA
                
                EMMA (Experimental Music Making Algorithm) is an AI-powered music generation and enhancement platform.
                
                ### Features:
                - **AI Lyrics Generation** - Generate lyrics from text prompts
                - **Music Generation** - Create 32-second clips with AI
                - **Stem Separation** - Separate vocals, drums, bass, and other instruments
                - **Audio Enhancement** - Professional mastering presets
                - **Vocal Processing** - Clean and enhance vocals
                - **Timeline System** - Arrange clips like a DAW
                - **Clip Library** - Organize and manage your creations
                
                ### Technology Stack:
                - ACE-Step for music generation
                - LyricsMindAI for lyrics generation
                - Demucs for stem separation
                - Pedalboard for audio processing
                - Gradio for the user interface
                
                ### License
                Apache 2.0 (or GPL 3.0 where required by dependencies)
                
                ### Credits
                Created by **Gamahea** as part of the **LEMM Project**
                
                *Making high-quality AI music production free and open source for all.*
                """)
        
        return app
    
    def launch(self, share: bool = False, server_port: int = 7860):
        """Launch the Gradio app"""
        import os
        
        # Set global instance for GPU-decorated functions
        global _ui_instance
        _ui_instance = self
        
        app = self.create_interface()
        
        # Auto-enable share for HuggingFace Spaces
        if os.getenv('SPACE_ID'):
            share = True
        
        logger.info(f"Launching EMMA UI on port {server_port}...")
        app.launch(
            share=share,
            server_port=server_port,
            server_name="0.0.0.0"
        )


# Global instance for GPU-decorated functions
_ui_instance = None

def _get_ui_instance():
    """Get the global UI instance"""
    global _ui_instance
    if _ui_instance is None:
        raise RuntimeError("UI not initialized")
    return _ui_instance


# GPU-decorated wrapper functions for HuggingFace Spaces ZeroGPU
if HAS_SPACES:
    import traceback
    import inspect
    # try to record call signatures to help debug runtime wrapper issues on hosted Spaces
    @spaces.GPU
    def gpu_generate_music(prompt, lyrics, timeline_position, auto_lyrics, duration):
        """GPU-accelerated music generation wrapper"""
        try:
            logger.debug(f"gpu_generate_music called (HAS_SPACES): args={prompt, lyrics, timeline_position, auto_lyrics, duration}")
            # log the function spec for sanity
            logger.debug(f"gpu_generate_music spec: {inspect.getfullargspec(gpu_generate_music)}")
            result = _get_ui_instance().generate_music(prompt, lyrics, timeline_position, auto_lyrics, duration)
            # Add empty error log on success
            if isinstance(result, tuple):
                return (*result, "",)
            else:
                return result, ""
        except Exception as e:
            tb = traceback.format_exc()
            return None, f"Error: {str(e)}", None, None, None, None, tb
    
    @spaces.GPU
    def gpu_apply_mastering(audio_input, preset):
        """GPU-accelerated audio mastering wrapper"""
        return _get_ui_instance().apply_mastering(audio_input, preset)
else:
    # No-op wrappers for local development
    import traceback
    import inspect
    def gpu_generate_music(prompt, lyrics, timeline_position, auto_lyrics, duration):
        try:
            logger.debug(f"gpu_generate_music called (local): args={prompt, lyrics, timeline_position, auto_lyrics, duration}")
            logger.debug(f"gpu_generate_music spec: {inspect.getfullargspec(gpu_generate_music)}")
            result = _get_ui_instance().generate_music(prompt, lyrics, timeline_position, auto_lyrics, duration)
            if isinstance(result, tuple):
                return (*result, "",)
            else:
                return result, ""
        except Exception as e:
            tb = traceback.format_exc()
            return None, f"Error: {str(e)}", None, None, None, None, tb
    
    def gpu_apply_mastering(audio_input, preset):
        return _get_ui_instance().apply_mastering(audio_input, preset)
