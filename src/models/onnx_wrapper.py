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

    def infer(self, prompt: str = None, lyrics: Optional[str] = None, duration: int = 32, reference_audio: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """Run inference on ONNX model.

        Note: This is a best-effort shim — the ONNX model must accept the
        inputs used here. If the model does not match, an informative error
        will be raised.
        """
        if not self.is_loaded or self.session is None:
            raise RuntimeError("ONNX model session not initialized. Call load() first and ensure model_path points to a valid ONNX file.")

        # Build inputs in a best-effort way. ONNX models vary widely so we try
        # to support simple numeric models for smoke testing (toy models) as
        # well as detect when text/token inputs are required and raise useful
        # errors so callers can add custom mapping logic.
        input_meta = self.session.get_inputs()
        input_names = [i.name for i in input_meta]

        # If the model expects tokenized inputs like 'input_ids', surface
        # an explicit error so the developer knows they must convert text->ids
        if any('input_ids' == n for n in input_names):
            raise NotImplementedError("ONNXModel inference requires tokenized 'input_ids'. Provide tokenized inputs or extend ONNXModel.infer().")

        # If the model expects a text/prompt input then this wrapper cannot
        # map arbitrary raw text without model-specific logic.
        if any(n in ('text', 'prompt', 'prompt_text') for n in input_names):
            raise NotImplementedError("ONNXModel found text-like input names; please adapt ONNXModel.infer() to map prompt->model input for your model.")

        # Otherwise assume the model is numeric and create zero-filled tensors
        # for each input, trying to respect static shapes. Dynamic shapes will
        # get small defaults and may be influenced by `duration`.
        feeds = {}
        for meta in input_meta:
            name = meta.name
            shape = []
            for dim in meta.shape:
                # ONNX may provide None or string for dynamic axes
                if isinstance(dim, int) and dim > 0:
                    shape.append(dim)
                else:
                    # Choose a sensible default for dynamic dimensions
                    # Use 1 for batch dims, and duration-derived sizes for sample dims
                    if len(shape) == 0:
                        shape.append(1)
                    else:
                        # Use duration to produce additional samples
                        shape.append(max(1, int(duration * 10)))

            # Create zeros of type float32
            feeds[name] = np.zeros(tuple(shape), dtype=np.float32)

        # Run the session
        outputs = self.session.run(None, feeds)

        if not outputs:
            raise RuntimeError("ONNX model returned no outputs")

        out0 = outputs[0]
        out = np.array(out0)

        # Normalize shape to (channels, samples) for downstream code. If the
        # output is 1D, replicate it to stereo. If it looks like (batch, L),
        # take first batch. If shape is (channels, samples) already, keep it.
        if out.ndim == 1:
            audio = np.stack([out, out], axis=0)
        elif out.ndim == 2:
            # Heuristic: if first dim is batch, reduce to first element
            if out.shape[0] == 1:
                vec = out[0]
                audio = np.stack([vec, vec], axis=0)
            elif out.shape[0] == 2:
                audio = out
            else:
                # Unknown layout; try transposing if that yields channels-first
                if out.shape[0] < out.shape[1]:
                    audio = out
                else:
                    audio = out.T
                    if audio.shape[0] != 2:
                        # fallback: make stereo by duplicating first row
                        audio = np.stack([audio[0], audio[0]], axis=0)
        else:
            # For higher dims, collapse last dimension
            flattened = out.reshape(out.shape[0], -1)
            if flattened.shape[0] == 2:
                audio = flattened
            else:
                audio = np.stack([flattened[0], flattened[0]], axis=0)

        return audio

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
