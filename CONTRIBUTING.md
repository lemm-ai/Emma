# Contributing to EMMA

Thank you for your interest in contributing to EMMA! We welcome contributions from everyone.

## 🎯 Project Goals

EMMA is part of the **LEMM Project** with the mission of **democratizing AI music** and making high-quality AI music production free and open source for all.

## 🤝 How to Contribute

### Reporting Issues

- Check if the issue already exists
- Provide detailed information:
  - Operating system and version
  - Python version
  - GPU type (NVIDIA/AMD/None)
  - Steps to reproduce
  - Expected vs actual behavior
  - Error messages and logs

### Suggesting Features

- Open an issue with `[Feature Request]` in the title
- Describe the feature and its benefits
- Explain how it fits with EMMA's goals

### Code Contributions

1. **Fork the Repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/emma.git
   cd emma
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Follow the existing code style
   - Add docstrings to functions and classes
   - Include type hints
   - Add logging where appropriate
   - Write comprehensive error handling

4. **Test Your Changes**
   - Ensure the code runs without errors
   - Test on both CPU and GPU if possible
   - Test the UI thoroughly

5. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "Add: description of your changes"
   ```

6. **Push to GitHub**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request**
   - Describe what you changed and why
   - Link to related issues
   - Include screenshots for UI changes

## 📋 Code Style

- Follow PEP 8 style guidelines
- Use type hints for function parameters and returns
- Write docstrings for all public functions and classes
- Keep functions focused and under 50 lines when possible
- Use meaningful variable names

### Example:

```python
def process_audio(
    audio: np.ndarray,
    sample_rate: int = 48000,
    normalize: bool = True
) -> np.ndarray:
    """
    Process audio with optional normalization
    
    Args:
        audio: Input audio as numpy array
        sample_rate: Sample rate in Hz
        normalize: Whether to normalize audio
        
    Returns:
        Processed audio array
        
    Raises:
        ValueError: If audio is invalid
    """
    if audio.size == 0:
        raise ValueError("Audio cannot be empty")
    
    # Processing logic here
    logger.info(f"Processing audio: {audio.shape}")
    
    return audio
```

## 🔍 Pull Request Guidelines

- Keep PRs focused on a single feature or fix
- Update documentation if needed
- Add your name to contributors
- Be responsive to feedback

## 🏗️ Development Setup

1. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

2. Set up pre-commit hooks (if using):
   ```bash
   pre-commit install
   ```

3. Run tests:
   ```bash
   pytest tests/
   ```

## 📚 Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions
- Comment complex logic
- Update config.yaml documentation

## 🐛 Debugging

- Use logging instead of print statements
- Check logs/emma.log for detailed information
- Enable DEBUG logging: set level to DEBUG in config.yaml

## 🎨 UI Contributions

- Test on different screen sizes
- Ensure accessibility
- Follow existing design patterns
- Include screenshots in PR

## 📝 License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License (or GPL 3.0 where required by dependencies).

## 💬 Communication

- Be respectful and constructive
- Ask questions if unclear
- Help other contributors

## 🌟 Recognition

All contributors will be acknowledged in the project. Your contributions help make AI music accessible to everyone!

---

Thank you for contributing to EMMA and the LEMM Project! 🎵
