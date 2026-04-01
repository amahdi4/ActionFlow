"""Evaluation helpers for ActionFlow classifiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from actionflow.training.metrics import (
    classification_report,
    compute_accuracy,
    compute_confusion_matrix,
    plot_confusion_matrix,
)


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: Sequence[str],
    device: str,
    output_dir: str | Path,
    mode: str = "flow",
) -> dict[str, Any]:
    """Evaluate a model, save metrics artifacts, and return the metric payload."""
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            logits = model(inputs.to(device))
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())

    accuracy = compute_accuracy(all_preds, all_labels)
    cm = compute_confusion_matrix(all_preds, all_labels, class_names)
    report = classification_report(all_preds, all_labels, class_names)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figure = plot_confusion_matrix(cm, class_names, output_path / f"confusion_matrix_{mode}.png")
    figure.savefig(output_path / "confusion_matrix.png")
    plt.close(figure)

    report_frame = pd.DataFrame(report).transpose()
    report_frame.to_csv(output_path / f"classification_report_{mode}.csv", index=True)

    metrics_payload = {
        "mode": mode,
        "accuracy": accuracy,
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    with (output_path / f"metrics_{mode}.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)
    with (output_path / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    print(f"Evaluation accuracy: {accuracy:.4f}")
    return metrics_payload
