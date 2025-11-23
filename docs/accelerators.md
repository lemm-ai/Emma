# Accelerators and GPU Backends (DirectML / ONNX / ROCm / CUDA)

This document explains how EMMA detects and uses accelerator backends and how to enable DirectML (ONNX Runtime) support for AMD GPUs (Windows) or ROCm (Linux) where available.

## Overview

EMMA now detects available backends at runtime and exposes a "Backend / Device" picker in the Generate UI. Supported backends include:

- cuda — NVIDIA GPUs (PyTorch CUDA)
- rocm — AMD ROCm (Linux server GPUs when ROCm/PyTorch ROCm builds are available)
- onnx_dml — ONNX Runtime using DirectML provider (Windows, AMD GPUs supported via DirectML)
- onnx — ONNX Runtime (CPU provider fallback)
- cpu — CPU-only inference

The codebase includes a small ONNX runtime wrapper (`src/models/onnx_wrapper.py`) and accelerator detection utilities (`src/utils/accelerator.py`). The ONNX wrapper is a general-purpose shim — it expects a compatible ONNX export of the model and must be adapted to your model's input/output schema.

## DirectML via ONNX (Windows)

DirectML is available on Windows through the ONNX Runtime DML provider. To use ONNX/DirectML with EMMA:

1. Export your model(s) to ONNX format (this depends on your model — e.g., task-specific conversion pipelines or using torch.onnx export).
2. Place the exported `.onnx` file and point `config.yaml` `model.ace_step_model_path` (or set `EMMA_DEVICE` env var) to the ONNX path.
3. Install ONNX Runtime with DirectML support on Windows. Typically, install the `onnxruntime-directml` package using pip (Windows only):

```powershell
pip install onnxruntime-directml
```

4. Select `onnx_dml` from the backend dropdown in the Generate Music UI.

Notes:
- ONNX conversion of large music/audio models can be non-trivial and may require custom serialization. The included ONNX wrapper is intentionally generic and will need to be adapted for the model's exact input schema.
- If `onnxruntime` is installed but the engine doesn't expose `DmlExecutionProvider`, the wrapper will fall back to CPU execution.

## ROCm (Linux)

ROCm is the AMD GPU stack for Linux. It supports PyTorch ROCm builds for server-class GPUs. If your GPU supports ROCm and you have a PyTorch ROCm build installed, setting the device to `rocm` will try to use PyTorch on ROCm.

## Why Vulkan is optional

Vulkan is a lower-level API and can be used for compute via specialized runtimes (e.g., ggml + Vulkan backends) but is not a drop-in replacement for PyTorch/ONNX workflows for large models. We recommend DirectML (Windows) and ROCm (Linux) as the first approach. Vulkan remains an advanced / optional path for quantized local inference.

## Next steps / TODOs

- Add sample ONNX conversion and small test model so ONNX runtime path is demonstrable.
- Add CI checks for detection code and low-cost inference smoke tests.
