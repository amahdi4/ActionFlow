from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(slots=True)
class FlowField:
    horizontal: Any = None
    vertical: Any = None
    magnitude: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class FlowEstimator(ABC):
    name = "base"

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        self.params = dict(params or {})

    @classmethod
    def is_available(cls) -> bool:
        return True

    @classmethod
    def availability_details(cls) -> str:
        return "No runtime requirements declared."

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.is_available(),
            "availability_details": self.availability_details(),
            "params": self.params,
        }

    @abstractmethod
    def estimate(self, frame_a: Any, frame_b: Any) -> FlowField:
        raise NotImplementedError

