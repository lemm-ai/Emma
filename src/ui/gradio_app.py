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
            result = self.music_gen.generate_music(
                prompt=prompt,
                lyrics=lyrics if lyrics else None,
                auto_generate_lyrics=auto_lyrics and not lyrics,
                separate_stems=True
            )
            
            self.current_clip = result['audio']
            self.current_stems = result['stems']
            
            # Add to timeline
            from ..timeline.manager import ClipPosition
            position_map: dict[str, ClipPosition] = {
                "Intro": "intro",
                "Previous": "previous",
                "Next": "next",
                "Outro": "outro"
            }
            
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
            
            return (settings.audio.sample_rate, result['audio']), f"Music generated successfully! Clip ID: {clip_id}"
            
        except Exception as e:
            logger.error(f"Error generating music: {e}")
            return None, f"Error: {str(e)}"
    
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
            processed = self.audio_enhancer.apply_mastering_preset(audio, preset.lower())
            
            return (sample_rate, processed), f"Applied {preset} mastering"
            
        except Exception as e:
            logger.error(f"Error applying mastering: {e}")
            return None, f"Error: {str(e)}"
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface"""
        
        with gr.Blocks(title="EMMA - Experimental Music Making Algorithm") as app:
            gr.Markdown("""
            # 🎵 EMMA - Experimental Music Making Algorithm
            
            AI-powered music generation and enhancement platform
            
            **Created by Gamahea / LEMM Project** | Making AI music free and open source for all
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
                
                # Connect buttons
                gen_lyrics_btn.click(
                    fn=self.generate_lyrics,
                    inputs=[prompt_input],
                    outputs=[lyrics_input]
                )
                
                generate_btn.click(
                    fn=self.generate_music,
                    inputs=[prompt_input, lyrics_input, timeline_position, auto_lyrics_check],
                    outputs=[audio_output, status_text]
                )
            
            with gr.Tab("🎚️ Audio Enhancement"):
                with gr.Row():
                    with gr.Column():
                        enhance_audio_input = gr.Audio(label="Audio Input")
                        
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
                        
                    with gr.Column():
                        enhanced_output = gr.Audio(label="Enhanced Audio")
                        enhance_status = gr.Textbox(label="Status", interactive=False)
                
                apply_mastering_btn.click(
                    fn=self.apply_mastering,
                    inputs=[enhance_audio_input, mastering_preset],
                    outputs=[enhanced_output, enhance_status]
                )
            
            with gr.Tab("📚 Clip Library"):
                gr.Markdown("### Your Generated Clips")
                gr.Markdown("*Clip library features will be available once models are integrated*")
                
                with gr.Row():
                    search_box = gr.Textbox(label="Search clips", placeholder="Search by name or prompt...")
                    search_btn = gr.Button("Search")
                
                clips_info = gr.Textbox(
                    label="Clips",
                    lines=10,
                    interactive=False,
                    placeholder="No clips yet. Generate music to see clips here."
                )
                
                with gr.Row():
                    clip_id_input = gr.Textbox(label="Clip ID")
                    load_clip_btn = gr.Button("Load to Timeline")
                    delete_clip_btn = gr.Button("Delete", variant="stop")
            
            with gr.Tab("⏱️ Timeline"):
                gr.Markdown("### Song Timeline")
                
                timeline_info = gr.Textbox(label="Timeline Info", lines=10, interactive=False)
                
                with gr.Row():
                    refresh_timeline_btn = gr.Button("Refresh Timeline")
                    clear_timeline_btn = gr.Button("Clear Timeline", variant="stop")
                    export_timeline_btn = gr.Button("Export Song", variant="primary")
                
                timeline_audio = gr.Audio(label="Timeline Preview")
                
                export_output = gr.File(label="Download")
            
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
