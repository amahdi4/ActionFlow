# ActionFlow

ActionFlow is a Python application for human action recognition driven by motion estimation.

Phase 1 establishes the project skeleton:

- modular package structure under `src/`
- config loading and default profiles
- device selection for `cuda`, `mps`, and `cpu`
- base interfaces for motion estimators, motion representations, and classifiers
- a dry-run pipeline for wiring validation
- a Gradio web UI scaffold with upload and webcam modes

## Planned pipeline

1. Read a video clip or webcam frames.
2. Compute motion with a selected backend.
3. Convert motion into a model-ready representation.
4. Run a classifier.
5. Show the predicted action and motion visualization.

## Environment

Recommended Python version: `3.11`.

The project is designed to work across:

- Windows or Linux with CUDA for training and heavy experiments
- macOS with `mps` when available
- CPU-only fallback for testing

## Install

Create a virtual environment and install the package in editable mode:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Install PyTorch using the wheel that matches your machine:

- Windows PC with NVIDIA GPU: install CUDA-enabled `torch` and `torchvision`
- macOS: install the standard macOS wheels with MPS support where available

## Launch

The canonical user-facing entrypoint is the installed `actionflow` command.

Launch the web app:

```bash
actionflow launch
```

The launch command defaults to the `demo` profile. Use `--device` to force a runtime target during manual testing:

```bash
actionflow --device cpu launch
actionflow --device mps launch
```

## Manual Testing

Inspect the resolved runtime environment:

```bash
actionflow check-env
```

Run a dry wiring check without opening the UI:

```bash
actionflow dry-run
```

Print the resolved configuration:

```bash
actionflow show-config
```

The demo opens at `http://127.0.0.1:7860` by default.

Inside the app:

- `Live Capture Lab` streams webcam frames continuously and overlays motion feedback in real time.
- `Clip Review` inspects uploaded videos and summarizes motion strength and scene quality.
- `How To Collect Better Data` gives a short checklist for recording useful action-recognition samples.

## Launch Behavior

`actionflow launch` is the real app entrypoint for manual use.

- `launch` opens a live motion-analysis workbench with the `demo` profile by default.
- `dry-run` validates configuration, device selection, and component wiring without starting the UI.
- `check-env` shows whether the runtime can use `cuda`, `mps`, or only `cpu`.

If you are working directly from source without installing the package, the module form still works as a development fallback:

```bash
PYTHONPATH=src python3 -m scripts.cli launch
```

## Current status

The repository currently contains scaffolding and interface contracts. Optical flow, training, and live prediction logic will be implemented in later phases.
