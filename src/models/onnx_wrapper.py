"""
ONNX Runtime model wrapper

Provides a simple ONNX-based model wrapper that can use ONNX Runtime providers
including DmlExecutionProvider (DirectML) on Windows. This is a best-effort shim
— users must provide an ONNX-exported model compatible with the expected inputs.
"""
from typing import Optional, Dict, Any
import logging
import numpy as np

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class ONNXModel(BaseModel):
    """Simple ONNX runtime model wrapper.

    This wrapper attempts to load an ONNX model and run inference using
    onnxruntime with an appropriate execution provider (DML/CUDA/CPU).
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "onnx"):
        # Do not call super to avoid torch device selection conflicts for ONNX
        self.model_path = model_path
        self.device = device
        self.session = None
        self.is_loaded = False
        self.use_fallback = False

    def load(self) -> None:
        try:
            import onnxruntime as ort

            providers = ort.get_available_providers()
            chosen = None

            # Prefer provider based on device hint
            if 'dml' in self.device.lower() or 'directml' in self.device.lower():
                if any('Dml' in p or 'DML' in p for p in providers):
                    chosen = [p for p in providers if 'Dml' in p or 'DML' in p]
            if not chosen and 'cuda' in self.device.lower():
                if 'CUDAExecutionProvider' in providers:
                    chosen = ['CUDAExecutionProvider']
            # Fallback to CPU
            if not chosen:
                chosen = ['CPUExecutionProvider']

            logger.info(f"ONNX Runtime providers available: {providers}. Using: {chosen}")

            so = ort.SessionOptions()
            # Keep default options for now

            if not self.model_path:
                raise RuntimeError("No ONNX model path provided. Set settings.model.ace_step_model_path to a valid .onnx file")

            self.session = ort.InferenceSession(self.model_path, sess_options=so, providers=chosen)
            self.is_loaded = True
            logger.info("ONNX model loaded into ONNX Runtime session")
        except Exception as e:
            logger.error(f"Failed to initialize ONNX Runtime session: {e}")
            raise

    def unload(self) -> None:
        if self.session is not None:
            try:
                # There's no explicit close API for session; allow GC
                self.session = None
                self.is_loaded = False
                logger.info("ONNX model session cleared")
            except Exception:
                pass

    def infer(self, prompt: str, lyrics: Optional[str] = None, duration: int = 32, reference_audio: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """Run inference on ONNX model.

        Note: This is a best-effort shim — the ONNX model must accept the
        inputs used here. If the model does not match, an informative error
        will be raised.
        """
        if not self.is_loaded or self.session is None:
            raise RuntimeError("ONNX model session not initialized. Call load() first and ensure model_path points to a valid ONNX file.")

        # Build inputs in a best-effort way. ONNX models vary widely; we try common
        # names like 'input_ids', 'prompt', 'text' — otherwise we raise a clear error.
        input_names = [i.name for i in self.session.get_inputs()]
        inputs = {}

        # Attempt to find a sensible text input field
        if 'input_ids' in input_names:
            # User must supply tokenized inputs — we can't do that generically here
            raise NotImplementedError("ONNXModel inference requires tokenized 'input_ids' input. Convert your model and provide a wrapper that tokenizes text to input_ids.")

        if 'text' in input_names or 'prompt' in input_names:
            field = 'text' if 'text' in input_names else 'prompt'
            # Many ONNX models expect arrays of bytes or ints; we cannot reliably convert here
            raise NotImplementedError("ONNX inference requires a validated input mapping. Please provide an ONNX model that accepts tokenized or numeric inputs, or extend ONNXModel.infer() to match your model's input schema.")

        # If no known mapping, raise an informative error
        raise RuntimeError(f"ONNX model inputs {input_names} are unknown. Provide a compatible ONNX export or extend ONNXModel.infer to map prompt/duration -> model inputs.")

    def get_clip_segments(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        sample_rate = 48000
        lead_in_samples = int(2 * sample_rate)
        lead_out_samples = int(2 * sample_rate)
        return {
            'lead_in': audio[:, :lead_in_samples],
            'core': audio[:, lead_in_samples:-lead_out_samples],
            'lead_out': audio[:, -lead_out_samples:],
            'full': audio
        }
