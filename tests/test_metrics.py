"""Tests for metric helpers."""

import numpy as np

from actionflow.training.metrics import classification_report, compute_accuracy, compute_confusion_matrix


def test_compute_accuracy_matches_expected_value() -> None:
    """Accuracy should reflect the fraction of correct predictions."""
    preds = [0, 1, 1, 2]
    labels = [0, 1, 0, 2]

    accuracy = compute_accuracy(preds, labels)

    assert accuracy == 0.75


def test_confusion_matrix_shape_matches_class_count() -> None:
    """Confusion matrices should be square over the configured classes."""
    class_names = ("boxing", "walking", "running")
    preds = [0, 1, 2, 1]
    labels = [0, 2, 2, 1]

    cm = compute_confusion_matrix(preds, labels, class_names)

    assert cm.shape == (3, 3)


def test_classification_report_contains_expected_keys() -> None:
    """Reports should include per-class metrics and summary entries."""
    class_names = ("boxing", "walking", "running")
    preds = np.array([0, 1, 2, 1])
    labels = np.array([0, 2, 2, 1])

    report = classification_report(preds, labels, class_names)

    assert "boxing" in report
    assert "walking" in report
    assert "running" in report
    assert "accuracy" in report
