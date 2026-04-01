"""Device selection helpers."""

from __future__ import annotations

import torch


def detect_best_device() -> str:
    """Return the best available torch device, preferring CUDA, then MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(preferred: str | None = None) -> str:
    """Resolve a user-selected device while defaulting to CPU."""
    if preferred in {None, ""}:
        return "cpu"
    if preferred == "auto":
        return detect_best_device()
    if preferred == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise ValueError("CUDA was requested but is not available.")
    if preferred == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        raise ValueError("MPS was requested but is not available.")
    if preferred == "cpu":
        return "cpu"
    if preferred.startswith("cuda:"):
        if torch.cuda.is_available():
            return preferred
        raise ValueError(f"{preferred} was requested but CUDA is not available.")
    raise ValueError(f"Unsupported device preference: {preferred}")
