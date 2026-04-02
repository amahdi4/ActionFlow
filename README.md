# ActionFlow

ActionFlow is a notebook-first project for human action recognition on the KTH Actions dataset.

The source of truth is [`ActionFlow.ipynb`](ActionFlow.ipynb). The notebook explains and runs the full workflow:

1. configuration
2. dataset check / download
3. frame extraction
4. dense Farneback optical flow
5. dataset split
6. three ResNet-18 experiments:
   optical flow, single-frame appearance, and temporal appearance
7. evaluation
8. comparison plots and saved artifacts

The Python package under `src/actionflow/` exists to support the notebook. If the notebook and helper modules ever drift, follow the notebook.

## What It Produces

Running the notebook writes artifacts such as:

```text
data/kth/raw/{class}/*.avi
data/kth/frames/{class}/{video_name}/frame_XXXXX.png
data/kth/flow/{class}/{video_name}/flow_XXXXX.npy

outputs/flow/best_flow.pt
outputs/flow/metrics_flow.json
outputs/appearance_single/best_appearance_single.pt
outputs/appearance_single/metrics_appearance_single.json
outputs/appearance_temporal/best_appearance_temporal.pt
outputs/appearance_temporal/metrics_appearance_temporal.json
outputs/comparison.json
```

## How To Run

Create a virtual environment, install the project with notebook dependencies, then open the notebook and run it top to bottom.

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[notebook]"
python -m notebook ActionFlow.ipynb
```

If you use VS Code, open [`ActionFlow.ipynb`](ActionFlow.ipynb) and select the project kernel or interpreter for `.venv`.

## Notes

- The notebook is cache-aware. If raw videos, frames, or flow files already exist, they are reused.
- Training uses CUDA automatically if your notebook kernel has a CUDA-enabled PyTorch install. Otherwise it falls back to CPU.
- The CLI and tests are secondary utilities, not the primary workflow.
