"""Synthetic end-to-end smoke tests."""

from pathlib import Path

from actionflow.cli import run_smoke_test


def test_smoke_pipeline_runs_end_to_end(tmp_path: Path) -> None:
    """The synthetic smoke test should train, evaluate, and write artifacts quickly."""
    result = run_smoke_test(output_dir=str(tmp_path))

    history = result["history"]

    assert len(history["train_loss"]) == 2
    assert history["train_loss"][-1] <= history["train_loss"][0]
    assert (tmp_path / "best_flow.pt").exists()
    assert (tmp_path / "training_curves.png").exists()
    assert (tmp_path / "confusion_matrix.png").exists()
    assert (tmp_path / "metrics.json").exists()
    assert result["elapsed_seconds"] < 60
