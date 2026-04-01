# ActionFlow Notes

## Project

ActionFlow is now notebook-first. The primary deliverable is:

- `ActionFlow.ipynb`

The important runtime contract is:

- raw data lives under `data/kth/raw/{class}/*.avi`
- prepared frames live under `data/kth/frames/...`
- prepared optical flow lives under `data/kth/flow/...`
- default runtime is `device="cpu"` and `num_workers=0`

## Main Entry Point

Open and run `ActionFlow.ipynb` top to bottom. The notebook owns:

- config
- dataset download via `download_kth.sh`
- frame extraction
- flow computation
- dataset loading
- orchestration
- inline plots

Only the model, trainer, and metrics helpers are intended to stay in `.py` modules.

## Secondary Commands

```bash
actionflow smoke-test
actionflow prepare-data
actionflow train --mode flow --subset 10 --epochs 2
actionflow train --mode rgb --subset 10 --epochs 2
actionflow evaluate --checkpoint outputs/best_flow.pt --mode flow
```

## Development Guardrails

- Keep tests CPU-only and synthetic unless a command explicitly targets real KTH data.
- Preserve type hints and docstrings on public APIs.
- Avoid circular imports between `data`, `training`, `evaluation`, and `utils`.
- Keep `num_workers=0` as the default for Mac-safe execution.
- Favor resumable preprocessing: do not recompute frames or flow that already exist.

## Validation

```bash
python -m json.tool ActionFlow.ipynb >/dev/null
PYTHONPATH=src python -m pytest tests/ -v
ruff check src/ tests/
actionflow smoke-test
```
