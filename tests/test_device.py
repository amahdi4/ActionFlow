"""Tests for runtime utility helpers."""

import torch

from actionflow.utils.device import detect_best_device, resolve_device
from actionflow.utils.seed import seed_everything


def test_resolve_device_defaults_to_cpu() -> None:
    """Device resolution should honor the CPU-first project default."""
    assert resolve_device() == "cpu"


def test_detect_best_device_returns_known_value() -> None:
    """Auto-detected devices should be one of the supported backends."""
    assert detect_best_device() in {"cpu", "cuda", "mps"}


def test_seed_everything_is_reproducible() -> None:
    """Repeated seeding should reproduce torch random tensors."""
    seed_everything(123)
    first = torch.rand(4)
    seed_everything(123)
    second = torch.rand(4)

    assert torch.equal(first, second)
