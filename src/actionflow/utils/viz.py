"""Visualization helpers for ActionFlow artifacts."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from actionflow.data.flow import visualize_flow
from actionflow.training.metrics import plot_confusion_matrix, plot_training_curves


def save_flow_visualization(flow_array: np.ndarray, save_path: str | Path) -> None:
    """Render an optical-flow tensor as an RGB PNG image."""
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = visualize_flow(flow_array)
    cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


__all__ = ["plot_confusion_matrix", "plot_training_curves", "save_flow_visualization"]
