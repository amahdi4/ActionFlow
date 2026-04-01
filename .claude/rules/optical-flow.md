---
paths:
  - "src/flow/**/*.py"
  - "src/representations/**/*.py"
---

- All flow estimators extend `flow.base.FlowEstimator` and return `FlowField` dataclasses
- Register new backends in `flow/__init__.py:FLOW_BACKENDS` dict
- Farneback and Lucas-Kanade require OpenCV; RAFT requires PyTorch+torchvision
- Representations registered in `representations/base.py:REPRESENTATIONS` dict
- stacked_flow = 2 channels per timestep; magnitude_map = 1 channel per timestep
- Config auto-corrects `model.input_channels` based on representation and clip_length
