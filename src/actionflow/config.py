"""Configuration objects for ActionFlow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ActionFlowConfig:
    """Dataclass-based runtime configuration for the ActionFlow pipeline."""

    data_root: str = "data/kth"
    clip_length: int = 10
    frame_stride: int = 2
    resize: tuple[int, int] = (224, 224)
    num_classes: int = 6
    class_names: tuple[str, ...] = ("boxing", "handclapping", "handwaving", "jogging", "running", "walking")

    flow_backend: str = "farneback"
    cache_flow: bool = True

    mode: str = "flow"
    input_channels: int = 20
    pretrained_backbone: bool = True

    batch_size: int = 16
    epochs: int = 25
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"

    device: str = "cpu"
    num_workers: int = 0
    seed: int = 42
    output_dir: str = "outputs"

    subset: int | None = None
    smoke_test: bool = False

    def __post_init__(self) -> None:
        """Validate config values and normalize derived fields."""
        if self.mode not in {"flow", "rgb"}:
            msg = f"Unsupported mode: {self.mode!r}. Expected 'flow' or 'rgb'."
            raise ValueError(msg)
        if self.clip_length <= 0:
            raise ValueError("clip_length must be positive.")
        if self.frame_stride <= 0:
            raise ValueError("frame_stride must be positive.")
        if len(self.resize) != 2 or self.resize[0] <= 0 or self.resize[1] <= 0:
            raise ValueError("resize must be a tuple of two positive integers.")
        if len(self.class_names) != self.num_classes:
            raise ValueError("num_classes must match len(class_names).")
        if self.num_workers != 0:
            raise ValueError("ActionFlow defaults to CPU-safe execution and requires num_workers=0.")
        if self.device == "":
            raise ValueError("device must not be empty.")

        self.input_channels = self.clip_length * 2 if self.mode == "flow" else 3
        self.data_root = str(Path(self.data_root))
        self.output_dir = str(Path(self.output_dir))
