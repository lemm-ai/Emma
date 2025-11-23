"""
Accelerator detection utilities

Detects available compute backends (CUDA, ROCm, ONNX Runtime + DirectML) and
provides helper functions for selecting a runtime backend.
"""
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def detect_backends() -> Dict[str, bool]:
    """Detect available accelerator backends on this machine.

    Returns a dict with keys: cuda, rocm, onnx, onnx_dml
    """
    backends = {
        'cuda': False,
        'rocm': False,
        'onnx': False,
        'onnx_dml': False
    }

    # PyTorch CUDA check
    try:
        import torch
        backends['cuda'] = torch.cuda.is_available()
        # ROCm detection: check for hip build
        backends['rocm'] = hasattr(torch.version, 'hip') and (torch.version.hip is not None)
    except Exception:
        logger.debug("PyTorch not available for CUDA/ROCm detection")

    # ONNX Runtime check
    try:
        import onnxruntime as ort
        backends['onnx'] = True
        providers = ort.get_available_providers()
        # DML (DirectML) uses the provider 'DmlExecutionProvider' on Windows
        backends['onnx_dml'] = any('Dml' in p or 'DML' in p for p in providers)
    except Exception:
        logger.debug("ONNX Runtime not available for detection")

    return backends


def choose_preferred_backend(prefer: str = None) -> str:
    """Choose the preferred backend string.

    prefer: optional string like 'cuda', 'rocm', 'onnx_dml', or 'cpu'.
    Returns the chosen backend name.
    """
    backends = detect_backends()

    if prefer:
        # If user preference is available, return it or fall back
        if prefer.lower() in ['cuda', 'rocm', 'onnx', 'onnx_dml', 'cpu']:
            requested = prefer.lower()
            if requested == 'cuda' and backends.get('cuda'):
                return 'cuda'
            if requested == 'rocm' and backends.get('rocm'):
                return 'rocm'
            if requested in ('onnx', 'onnx_dml') and backends.get('onnx'):
                return 'onnx_dml' if backends.get('onnx_dml') else 'onnx'
            if requested == 'cpu':
                return 'cpu'

    # Auto-pick:
    if backends.get('cuda'):
        return 'cuda'
    if backends.get('onnx_dml'):
        return 'onnx_dml'
    if backends.get('rocm'):
        return 'rocm'
    if backends.get('onnx'):
        return 'onnx'
    return 'cpu'
