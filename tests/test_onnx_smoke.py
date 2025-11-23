import os
import sys
import tempfile
import numpy as np

from src.models.onnx_wrapper import ONNXModel


def test_onnx_toy_model_roundtrip(tmp_path):
    # Export toy model to a temp location
    models_dir = tmp_path / 'models'
    models_dir.mkdir()
    model_path = str(models_dir / 'toy_model.onnx')

    # Use the tool to create a toy model
    import tools.export_toy_onnx as exporter
    exporter.export(path=model_path, in_len=16, out_len=256)

    # Load into ONNXModel wrapper
    m = ONNXModel(model_path=model_path, device='onnx')
    m.load()

    # Call infer with a short duration -> should return a numpy audio-like array
    out = m.infer(duration=1)
    assert isinstance(out, np.ndarray)
    assert out.ndim == 2, f"Expected 2D (channels,samples), got {out.shape}"
    assert out.shape[0] in (1, 2) or out.shape[0] >= 1
