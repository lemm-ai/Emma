# HuggingFace Space Deployment

## Space URL
https://huggingface.co/spaces/Gamahea/Emma

## Deployment Status
✅ Successfully pushed to HuggingFace Space!

## Files Deployed
All project files have been pushed to the HuggingFace Space, including:
- Complete source code (src/)
- Main application (app.py)
- Configuration files (config.yaml, requirements.txt)
- Documentation (README with HF frontmatter)
- Setup scripts

## Important Notes

### README Files
- **README.md** (local) - Full documentation for development
- **README_HF.md** - HuggingFace Space version with YAML frontmatter
- **README_FULL.md** - Backup of full documentation

The HuggingFace Space uses README_HF.md (with proper YAML frontmatter for Spaces).

### Git Remotes
- `space` - HuggingFace Space remote (https://huggingface.co/spaces/Gamahea/Emma)

### To Update the Space
```powershell
# Make your changes
git add .
git commit -m "Your commit message"

# Push to HuggingFace Space
git push space master:main
```

### Space Configuration (from README frontmatter)
```yaml
title: EMMA - AI Music Generator
emoji: 🎵
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
```

## Next Steps for Production Deployment

1. **Download AI Models**
   - Models need to be available in the Space
   - Consider using HuggingFace model hub for model storage
   - Update model paths in config.yaml

2. **Update Requirements**
   - Ensure all dependencies are compatible with HF Spaces
   - May need to pin specific versions

3. **Configure Secrets**
   - Add environment variables in Space settings if needed
   - Use HF Secrets for sensitive configuration

4. **Test the Space**
   - Visit https://huggingface.co/spaces/Gamahea/Emma
   - The Space will build automatically
   - Check build logs for any errors

5. **Optimize for Spaces**
   - Consider file size limits
   - Optimize model loading
   - Add proper error handling for Space environment

## Model Integration Notes

The current deployment is a framework. For full functionality:

1. Download model weights:
   - ACE-Step
   - LyricsMindAI
   - Demucs (can auto-download)
   - So-VITS-SVC
   - MusicControlNet
   - AudioSR

2. Upload models to HuggingFace Hub or include in Space

3. Update model paths in config.yaml

4. Test each feature in the Space environment

## Troubleshooting

### Space Not Building
- Check requirements.txt for incompatible packages
- Review build logs in Space settings
- Ensure README has proper YAML frontmatter

### Runtime Errors
- Check Space logs
- Verify model paths are correct
- Ensure dependencies are installed

### ONNX / DirectML Notes
If you want to enable ONNX Runtime DirectML (recommended for AMD GPUs on Windows):

- Install the DirectML-enabled ONNX Runtime package locally or in your deployment environment:

```powershell
pip install onnxruntime-directml
```

- Make sure `config.yaml` points `model.ace_step_model_path` to a compatible ONNX export and select `onnx_dml` in the UI backend dropdown.


### Out of Memory
- HF Spaces have memory limits
- May need to optimize model loading
- Consider using smaller models or CPU-only versions

---

**Deployed:** November 21, 2025
**Status:** Framework deployed, awaiting model integration
