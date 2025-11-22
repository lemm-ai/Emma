# EMMA - Quick Reference Guide

## Quick Start Commands

### Windows (PowerShell)
```powershell
# Setup (first time only)
.\setup.ps1

# Or manually:
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Run EMMA
python app.py
```

### Linux/Mac
```bash
# Setup (first time only)
chmod +x setup.sh
./setup.sh

# Or manually:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run EMMA
python app.py
```

## Directory Structure

```
Emma/
├── .github/              # GitHub configuration
│   └── copilot-instructions.md
├── src/                  # Source code
│   ├── models/          # AI model wrappers
│   ├── music_generation/ # Music generation pipeline
│   ├── audio_processing/ # Audio enhancement
│   ├── timeline/        # Timeline management
│   ├── clip_library/    # Database management
│   ├── ui/              # Gradio interface
│   ├── config/          # Configuration
│   └── utils/           # Utilities
├── data/                # Database and user data
├── logs/                # Log files
├── output/              # Generated audio files
├── models/              # AI model weights (download separately)
├── app.py               # Main entry point
├── config.yaml          # Configuration file
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variables template
└── README.md           # Full documentation
```

## Key Files

- **app.py** - Main application entry point
- **config.yaml** - Configuration settings
- **.env** - Environment variables (create from .env.example)
- **requirements.txt** - Python package dependencies

## Configuration

### config.yaml
```yaml
model:
  device: "cuda"  # cuda, rocm, or cpu
  use_gpu: true

audio:
  sample_rate: 48000
  output_format: "wav"

database:
  db_type: "sqlite"
  sqlite_path: "data/emma_clips.db"

ui:
  server_port: 7860
  share: false
```

### .env
```bash
EMMA_DEVICE=cuda
EMMA_USE_GPU=true
EMMA_SERVER_PORT=7860
EMMA_SHARE=false
```

## Common Tasks

### Generate Music
1. Open http://localhost:7860
2. Enter a prompt (e.g., "upbeat pop song")
3. Optionally enter lyrics or enable auto-generation
4. Choose timeline position
5. Click "Generate Music"

### Apply Mastering
1. Go to "Audio Enhancement" tab
2. Upload or use generated audio
3. Select a mastering preset
4. Click "Apply Mastering"

### Manage Timeline
1. Go to "Timeline" tab
2. View all clips on timeline
3. Export final song

### Browse Clips
1. Go to "Clip Library" tab
2. Search or browse clips
3. Load, download, or delete clips

## Mastering Presets

- **Balanced** - All-purpose
- **Bright** - Enhanced highs
- **Warm** - Smooth tones
- **Punchy** - Strong impact
- **Radio** - Radio-ready
- **Streaming** - Platform optimized
- And 10 more...

## Troubleshooting

### GPU Not Detected
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check device info
python -c "from src.utils.device_utils import get_device_info; print(get_device_info())"
```

### Port Already in Use
Edit config.yaml and change server_port to a different port (e.g., 7861)

### Models Not Found
Download model weights and place in models/ directory. See README.md for links.

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## Development

### Project Structure
- Modular design with clear separation of concerns
- Base model class for all AI models
- Comprehensive logging and error handling
- Type hints throughout codebase

### Adding Features
1. Create module in appropriate src/ subdirectory
2. Follow existing code patterns
3. Add logging and error handling
4. Update documentation

### Testing
```bash
# Run application in debug mode
python app.py

# Check logs
cat logs/emma.log
```

## Resources

- **GitHub**: [Repository URL]
- **Models**: See README.md for download links
- **Issues**: Report bugs on GitHub
- **Contributing**: See CONTRIBUTING.md

## License

Apache 2.0 (or GPL 3.0 where required by dependencies)

---

**Made by Gamahea / LEMM Project**
*Democratizing AI Music for Everyone*
