"""
Export a tiny toy ONNX model for smoke-testing the ONNX/DirectML path.

This script creates a small PyTorch model that maps a 1D input vector to a
short output vector and exports it to ONNX at `models/toy_model.onnx`.

Run locally (requires torch) to create the onnx file for testing.
"""
import os
import torch
import torch.nn as nn


class ToySynth(nn.Module):
    def __init__(self, in_len=128, out_len=16000):
        super().__init__()
        self.fc = nn.Linear(in_len, out_len)

    def forward(self, x):
        # x shape: (batch, in_len)
        return self.fc(x)


def export(path: str = 'models/toy_model.onnx', in_len: int = 128, out_len: int = 4096):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model = ToySynth(in_len=in_len, out_len=out_len)
    model.eval()

    dummy = torch.zeros((1, in_len), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        path,
        export_params=True,
        opset_version=13,
        input_names=['input_0'],
        output_names=['output_0'],
        dynamic_axes={'input_0': {0: 'batch'}, 'output_0': {0: 'batch'}}
    )

    print(f"Exported toy ONNX model to: {path}")


if __name__ == '__main__':
    export()
