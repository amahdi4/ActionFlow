from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RepresentationSpec:
    name: str
    description: str
    channels_per_step: int


class MotionRepresentation(ABC):
    def __init__(self, spec: RepresentationSpec) -> None:
        self.spec = spec

    def summary(self) -> dict[str, object]:
        return {
            "name": self.spec.name,
            "description": self.spec.description,
            "channels_per_step": self.spec.channels_per_step,
        }

    @abstractmethod
    def transform(self, flow_sequence: Any) -> Any:
        raise NotImplementedError


class StackedFlowRepresentation(MotionRepresentation):
    def __init__(self) -> None:
        super().__init__(
            RepresentationSpec(
                name="stacked_flow",
                description="Concatenates horizontal and vertical flow channels across time.",
                channels_per_step=2,
            )
        )

    def transform(self, flow_sequence: Any) -> Any:
        raise NotImplementedError("Stacked flow tensors are scheduled for Phase 2.")


class MagnitudeMapRepresentation(MotionRepresentation):
    def __init__(self) -> None:
        super().__init__(
            RepresentationSpec(
                name="magnitude_map",
                description="Uses motion magnitude as a compact single-channel representation.",
                channels_per_step=1,
            )
        )

    def transform(self, flow_sequence: Any) -> Any:
        raise NotImplementedError("Magnitude map conversion is scheduled for Phase 2.")


REPRESENTATIONS = {
    "stacked_flow": StackedFlowRepresentation,
    "magnitude_map": MagnitudeMapRepresentation,
}


def build_representation(name: str) -> MotionRepresentation:
    normalized = name.lower()
    if normalized not in REPRESENTATIONS:
        choices = ", ".join(sorted(REPRESENTATIONS))
        raise ValueError(f"Unknown motion representation: {name}. Expected one of: {choices}")
    return REPRESENTATIONS[normalized]()

