# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is ActionFlow

ActionFlow is a motion-estimation-first human action recognition application. The pipeline reads video, computes optical flow, converts flow into a model-ready representation, and runs a classifier. Currently in Phase 1 (scaffolding and interface contracts); flow computation, training, and inference are stubbed.

## Commands

```bash
# Install (editable, with dev deps)
pip install -e .[dev]

# Install with training deps (PyTorch)
pip install -e .[dev,training]

# Run all tests
PYTHONPATH=src python -m pytest tests/

# Run a single test file
PYTHONPATH=src python -m pytest tests/test_config_loader.py

# Lint
ruff check src/ tests/

# CLI commands (all require PYTHONPATH=src)
PYTHONPATH=src python -m scripts.cli show-config
PYTHONPATH=src python -m scripts.cli dry-run
PYTHONPATH=src python -m scripts.cli launch-demo
```

## Architecture

All source lives under `src/` with `PYTHONPATH=src` (configured in `pyproject.toml` via `tool.setuptools.package-dir` and `tool.pytest.ini_options.pythonpath`). Imports use bare package names (e.g., `from configs.loader import load_config`), not `src.configs.loader`.

### Pipeline flow

`scripts/cli.py` → parses args → loads config → dispatches to:
- `training/pipeline.py` (`TrainingPipeline`) — wires together flow estimator, representation, classifier, and device selection
- `demo/app.py` — Gradio web UI scaffold

### Key abstractions (all ABC-based with `is_available()` + `summary()` pattern)

- **`flow/base.py: FlowEstimator`** → backends: `farneback` (OpenCV), `lucas_kanade` (OpenCV), `raft` (PyTorch). Registry in `flow/__init__.py: FLOW_BACKENDS`. Factory: `build_flow_estimator(name, params)`.
- **`representations/base.py: MotionRepresentation`** → `stacked_flow` (2ch/step), `magnitude_map` (1ch/step). Registry: `REPRESENTATIONS`. Factory: `build_representation(name)`.
- **`models/base.py: ClassifierBackend`** → currently only `resnet18_flow`. Factory in `models/factory.py: build_classifier(model_config, data_config)`.

Each subsystem follows the same pattern: an ABC defines the interface, concrete classes live in sibling modules, a dict registry maps string names to classes, and a `build_*` factory function does lookup + instantiation.

### Configuration

`configs/schema.py` defines the full config as nested dataclasses (`AppConfig` → `ProjectConfig`, `RuntimeConfig`, `DataConfig`, `FlowConfig`, `ModelConfig`, `TrainingConfig`, `EvaluationConfig`, `DemoConfig`). `AppConfig.from_mapping()` auto-corrects `num_classes` and `input_channels` to match `class_names` and `clip_length`. Default profiles live in `configs/defaults/` as JSON files (`train.json`, `demo.json`). Configs support both JSON and YAML.

### Device selection

`utils/device.py: detect_device()` probes for `cuda` → `mps` → `cpu` (in that order when `"auto"`). PyTorch is an optional dependency; when absent, falls back to CPU gracefully.
