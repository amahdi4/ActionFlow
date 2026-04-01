# ActionFlow

ActionFlow is a notebook-first KTH Actions project for optical-flow-based human action recognition. The main thing to run is `ActionFlow.ipynb`, which walks through the full pipeline top to bottom:

1. configuration
2. cache-aware frame extraction
3. dense Farneback optical flow
4. dataset loading
5. ResNet18 model construction
6. training
7. evaluation
8. inline visualizations

The notebook is written so a reader can understand the project just by reading it. Heavy reusable pieces stay in a few Python modules:

- `src/actionflow/models/resnet_flow.py`
- `src/actionflow/training/trainer.py`
- `src/actionflow/training/metrics.py`

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,notebook]'
```

## Data Layout

The notebook uses `download_kth.sh` to populate the canonical raw-data location:

```text
data/kth/raw/{class}/*.avi
```

If videos are already present there, the script skips them. If a class is missing, the script downloads only the missing class archive.

Prepared artifacts are written to:

```text
data/kth/frames/{class}/{video_name}/frame_XXXXX.png
data/kth/flow/{class}/{video_name}/flow_XXXXX.npy
```

## Run

Open and run `ActionFlow.ipynb` from top to bottom.

The notebook is cache-aware:

- it runs `download_kth.sh` first, which skips already downloaded classes
- if frames already exist, extraction is skipped
- if optical-flow files already exist, flow computation is skipped

By default the notebook uses a CPU-friendly per-class subset so it is practical on a laptop. Increase those limits in the config cell for a larger experiment.

## Secondary Utilities

The CLI and tests still exist, but they are secondary now:

```bash
actionflow smoke-test
actionflow prepare-data
actionflow train --mode flow --subset 10 --epochs 2
actionflow evaluate --checkpoint outputs/best_flow.pt --mode flow
```

```bash
PYTHONPATH=src python -m pytest tests/ -v
ruff check src/ tests/
```
