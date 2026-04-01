from flow.base import FlowEstimator, FlowField
from flow.farneback import FarnebackEstimator
from flow.lucas_kanade import LucasKanadeEstimator
from flow.raft import RaftEstimator

FLOW_BACKENDS = {
    "farneback": FarnebackEstimator,
    "lucas_kanade": LucasKanadeEstimator,
    "raft": RaftEstimator,
}


def build_flow_estimator(name: str, params: dict | None = None) -> FlowEstimator:
    normalized = name.lower()
    if normalized not in FLOW_BACKENDS:
        choices = ", ".join(sorted(FLOW_BACKENDS))
        raise ValueError(f"Unknown flow backend: {name}. Expected one of: {choices}")
    return FLOW_BACKENDS[normalized](params=params)


__all__ = ["FLOW_BACKENDS", "FlowEstimator", "FlowField", "build_flow_estimator"]

