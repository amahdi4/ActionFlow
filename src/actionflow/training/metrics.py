"""Metrics and plots for training and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report as sklearn_classification_report
from sklearn.metrics import confusion_matrix


def compute_accuracy(preds: Sequence[int] | np.ndarray | torch.Tensor, labels: Sequence[int] | np.ndarray | torch.Tensor) -> float:
    """Compute classification accuracy from predicted and true class ids."""
    pred_ids = _to_class_ids(preds)
    label_ids = _to_class_ids(labels)
    if pred_ids.shape != label_ids.shape:
        raise ValueError("preds and labels must have the same shape after conversion.")
    return float((pred_ids == label_ids).mean())


def compute_confusion_matrix(
    preds: Sequence[int] | np.ndarray | torch.Tensor,
    labels: Sequence[int] | np.ndarray | torch.Tensor,
    class_names: Sequence[str],
) -> np.ndarray:
    """Compute a confusion matrix using the configured class ordering."""
    pred_ids = _to_class_ids(preds)
    label_ids = _to_class_ids(labels)
    indices = list(range(len(class_names)))
    return confusion_matrix(label_ids, pred_ids, labels=indices)


def classification_report(
    preds: Sequence[int] | np.ndarray | torch.Tensor,
    labels: Sequence[int] | np.ndarray | torch.Tensor,
    class_names: Sequence[str],
) -> dict[str, dict[str, float] | float]:
    """Return a sklearn-style classification report as a dictionary."""
    pred_ids = _to_class_ids(preds)
    label_ids = _to_class_ids(labels)
    return sklearn_classification_report(
        label_ids,
        pred_ids,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        zero_division=0,
        output_dict=True,
    )


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Render a confusion matrix heatmap and optionally save it."""
    fig, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axis)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    fig.tight_layout()

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)

    return fig


def plot_training_curves(
    history: dict[str, list[float]],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot training and validation loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)

    return fig


def _to_class_ids(values: Sequence[int] | np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert logits or class ids into a flat numpy integer array."""
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)

    if array.ndim == 2:
        return array.argmax(axis=1).astype(int, copy=False)
    return array.astype(int, copy=False).reshape(-1)
