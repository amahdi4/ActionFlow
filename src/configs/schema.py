from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(slots=True)
class ProjectConfig:
    name: str = "ActionFlow"
    output_dir: str = "outputs"
    seed: int = 42


@dataclass(slots=True)
class RuntimeConfig:
    device: str = "auto"
    num_workers: int = 4
    mixed_precision: bool = True
    log_level: str = "INFO"


@dataclass(slots=True)
class DataConfig:
    dataset_name: str = "ucf101"
    dataset_root: str = "data/UCF101"
    split: str = "split_1"
    clip_length: int = 10
    frame_stride: int = 2
    resize_height: int = 224
    resize_width: int = 224
    class_names: list[str] = field(
        default_factory=lambda: [
            "running",
            "walking",
            "jumping",
            "falling",
        ]
    )
    cache_dir: str = "data/cache"


@dataclass(slots=True)
class FlowConfig:
    backend: str = "farneback"
    representation: str = "stacked_flow"
    precompute: bool = True
    cache_dir: str = "data/cache/flow"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelConfig:
    architecture: str = "resnet18_flow"
    num_classes: int = 4
    input_channels: int = 20
    pretrained: bool = False
    checkpoint: str | None = None


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 8
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


@dataclass(slots=True)
class EvaluationConfig:
    benchmark_batch_size: int = 1
    warmup_steps: int = 5
    timed_steps: int = 20
    measure_memory: bool = True


@dataclass(slots=True)
class DemoConfig:
    host: str = "127.0.0.1"
    port: int = 7860
    share: bool = False
    title: str = "ActionFlow Demo"
    enabled_backends: list[str] = field(
        default_factory=lambda: ["farneback", "lucas_kanade", "raft"]
    )
    live_frame_interval_ms: int = 100


@dataclass(slots=True)
class AppConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    demo: DemoConfig = field(default_factory=DemoConfig)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AppConfig":
        config = cls(
            project=ProjectConfig(**payload.get("project", {})),
            runtime=RuntimeConfig(**payload.get("runtime", {})),
            data=DataConfig(**payload.get("data", {})),
            flow=FlowConfig(**payload.get("flow", {})),
            model=ModelConfig(**payload.get("model", {})),
            training=TrainingConfig(**payload.get("training", {})),
            evaluation=EvaluationConfig(**payload.get("evaluation", {})),
            demo=DemoConfig(**payload.get("demo", {})),
        )

        if config.model.num_classes != len(config.data.class_names):
            config.model.num_classes = len(config.data.class_names)

        expected_channels = config.data.clip_length * 2
        if config.flow.representation == "magnitude_map":
            expected_channels = config.data.clip_length
        if config.model.input_channels != expected_channels:
            config.model.input_channels = expected_channels

        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
