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
        auto_lyrics: bool
    ):
        """Generate music clip"""
        try:
            if not prompt:
                return None, "Please enter a prompt first."
            
            # Generate music
            # Note: Disable stem separation on HF Spaces to avoid errors
            import os
            separate_stems = not os.getenv('SPACE_ID')
            
            result = self.music_gen.generate_music(
                prompt=prompt,
                lyrics=lyrics if lyrics else None,
                auto_generate_lyrics=auto_lyrics and not lyrics,
                separate_stems=separate_stems
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
            timeline_info = self.get_timeline_info()
            timeline_clips = self.get_timeline_clips_dataframe()
            
            return (
                (settings.audio.sample_rate, audio_for_gradio),  # audio_output
                f"Music generated successfully! Clip ID: {clip_id}",  # status_text
                (settings.audio.sample_rate, audio_for_gradio),  # enhance_audio_input
                timeline_info,  # timeline_info
                clips_info,  # clips_dataframe
                timeline_clips  # timeline_clips_df
            )
            
        except Exception as e:
            logger.error(f"Error generating music: {e}")
            return (
                None, 
                f"Error: {str(e)}", 
                None, 
                self.get_timeline_info(), 
                self.get_clip_library_info(),
                self.get_timeline_clips_dataframe()
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
    
    def get_timeline_info(self) -> str:
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
                return self.get_timeline_info(), self.get_timeline_clips_dataframe(), "No file uploaded"
            
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
                self.get_timeline_info(),
                self.get_timeline_clips_dataframe(),
                f"Successfully added '{filename}' to timeline and library!"
            )
            
        except Exception as e:
            logger.error(f"Error uploading to timeline: {e}")
            return (
                self.get_timeline_info(),
                self.get_timeline_clips_dataframe(),
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
        
        with gr.Blocks(title="EMMA - Experimental Music Making Algorithm") as app:
            gr.Markdown(f"""
            # 🎵 EMMA - Experimental Music Making Algorithm
            
            AI-powered music generation and enhancement platform
            
            **Created by Gamahea / LEMM Project** | Making AI music free and open source for all
            {fallback_warning}
            """)
            
            with gr.Tab("🎼 Generate Music"):
                with gr.Row():
                    with gr.Column(scale=2):
                        prompt_input = gr.Textbox(
                            label="Music Prompt",
                            placeholder="Describe the music you want to create (e.g., 'upbeat pop song with catchy melody')",
                            lines=3
                        )
                        
                        with gr.Row():
                            auto_lyrics_check = gr.Checkbox(
                                label="Auto-generate Lyrics",
                                value=True
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
                        
                        generate_btn = gr.Button("🎵 Generate Music", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Output")
                        audio_output = gr.Audio(label="Generated Music")
                        status_text = gr.Textbox(label="Status", interactive=False)
            
            with gr.Tab("🎚️ Audio Enhancement"):
                with gr.Row():
                    with gr.Column():
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
                        
                    with gr.Column():
                        enhanced_output = gr.Audio(label="Enhanced Audio")
                        enhance_status = gr.Textbox(label="Status", interactive=False)
            
            with gr.Tab("📚 Clip Library"):
                gr.Markdown("### Your Generated Clips")
                gr.Markdown("*Click a row to select, then use 'Load to Timeline' button*")
                
                with gr.Row():
                    search_box = gr.Textbox(label="Search clips", placeholder="Search by name or prompt...")
                    search_btn = gr.Button("Search")
                
                clips_dataframe = gr.Dataframe(
                    headers=["ID", "Name", "Prompt", "Duration", "Created"],
                    datatype=["str", "str", "str", "str", "str"],
                    label="Clips",
                    interactive=False,
                    wrap=True
                )
                
                with gr.Row():
                    load_clip_btn = gr.Button("Load to Timeline", variant="primary")
                    delete_clip_btn = gr.Button("Delete", variant="stop")
                    
                load_status = gr.Textbox(label="Status", interactive=False, visible=False)
            
            with gr.Tab("⏱️ Timeline"):
                gr.Markdown("### DAW Timeline - Arrange Your Clips")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        # Timeline visualization (using DataFrame for now, can be enhanced with custom HTML/JS)
                        timeline_clips_df = gr.Dataframe(
                            headers=["Track", "Clip Name", "Start (s)", "Duration (s)", "End (s)"],
                            datatype=["str", "str", "number", "number", "number"],
                            label="Timeline Clips",
                            interactive=False,
                            wrap=True,
                            row_count=10
                        )
                        
                        timeline_info = gr.Textbox(label="Timeline Info", lines=5, interactive=False)
                        
                    with gr.Column(scale=1):
                        gr.Markdown("### Controls")
                        
                        # Upload audio to add to timeline
                        upload_audio = gr.Audio(label="Upload Audio File", type="filepath")
                        upload_to_timeline_btn = gr.Button("➕ Add Upload to Timeline", variant="secondary")
                        
                        refresh_timeline_btn = gr.Button("🔄 Refresh Timeline")
                        clear_timeline_btn = gr.Button("🗑️ Clear Timeline", variant="stop")
                        
                        gr.Markdown("---")
                        
                        export_timeline_btn = gr.Button("📥 Export Song", variant="primary", size="lg")
                        export_output = gr.File(label="Download", interactive=False, file_count="single")
                
                timeline_audio = gr.Audio(label="Timeline Preview (Full Song)")
                upload_status = gr.Textbox(label="Status", interactive=False, visible=False)
            
            # Connect button callbacks after all components are defined
            gen_lyrics_btn.click(
                fn=self.generate_lyrics,
                inputs=[prompt_input],
                outputs=[lyrics_input]
            )
            
            generate_btn.click(
                fn=gpu_generate_music,
                inputs=[prompt_input, lyrics_input, timeline_position, auto_lyrics_check],
                outputs=[audio_output, status_text, enhance_audio_input, timeline_info, clips_dataframe, timeline_clips_df]
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
            
            upload_to_timeline_btn.click(
                fn=self.upload_to_timeline,
                inputs=[upload_audio],
                outputs=[timeline_info, timeline_clips_df, upload_status]
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
    @spaces.GPU
    def gpu_generate_music(prompt, lyrics, timeline_position, auto_lyrics):
        """GPU-accelerated music generation wrapper"""
        return _get_ui_instance().generate_music(prompt, lyrics, timeline_position, auto_lyrics)
    
    @spaces.GPU
    def gpu_apply_mastering(audio_input, preset):
        """GPU-accelerated audio mastering wrapper"""
        return _get_ui_instance().apply_mastering(audio_input, preset)
else:
    # No-op wrappers for local development
    def gpu_generate_music(prompt, lyrics, timeline_position, auto_lyrics):
        return _get_ui_instance().generate_music(prompt, lyrics, timeline_position, auto_lyrics)
    
    def gpu_apply_mastering(audio_input, preset):
        return _get_ui_instance().apply_mastering(audio_input, preset)
