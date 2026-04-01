from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ModelSummary:
    architecture: str
    num_classes: int
    input_channels: int
    pretrained: bool = False
    checkpoint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PredictionResult:
    label: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ClassifierBackend(ABC):
    def __init__(self, summary: ModelSummary) -> None:
        self.model_summary = summary

    @classmethod
    def is_available(cls) -> bool:
        return True

    @classmethod
    def availability_details(cls) -> str:
        return "No runtime requirements declared."

    def summary(self) -> dict[str, Any]:
        data = {
            "available": self.is_available(),
            "availability_details": self.availability_details(),
        }
        data.update(self.model_summary.metadata)
        data["architecture"] = self.model_summary.architecture
        data["num_classes"] = self.model_summary.num_classes
        data["input_channels"] = self.model_summary.input_channels
        data["pretrained"] = self.model_summary.pretrained
        data["checkpoint"] = self.model_summary.checkpoint
        return data

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, batch: Any) -> PredictionResult:
        raise NotImplementedError

