from __future__ import annotations

from typing import Any

from flow.base import FlowEstimator, FlowField


class RaftEstimator(FlowEstimator):
    name = "raft"

    @classmethod
    def is_available(cls) -> bool:
        return False

    @classmethod
    def availability_details(cls) -> str:
        return (
            "Torchvision ships RAFT models, but this demo keeps the backend disabled until pretrained "
            "weights and GPU execution are wired end-to-end."
        )

    def estimate(self, frame_a: Any, frame_b: Any) -> FlowField:
        raise NotImplementedError("RAFT integration is scheduled for Phase 4.")
