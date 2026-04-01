"""Tests for optical-flow helpers."""

import numpy as np

from actionflow.data.flow import compute_flow, visualize_flow


def test_compute_flow_returns_dense_field() -> None:
    """Farneback flow should return a finite dense vector field."""
    frame_a = np.zeros((64, 64), dtype=np.uint8)
    frame_b = np.zeros((64, 64), dtype=np.uint8)
    frame_a[16:32, 16:32] = 255
    frame_b[16:32, 18:34] = 255

    flow = compute_flow(frame_a, frame_b)

    assert flow.shape == (64, 64, 2)
    assert np.isfinite(flow).all()


def test_visualize_flow_returns_rgb_image() -> None:
    """Flow visualization should produce an RGB uint8 image."""
    flow = np.zeros((32, 32, 2), dtype=np.float32)
    flow[..., 0] = 1.0

    image = visualize_flow(flow)

    assert image.shape == (32, 32, 3)
    assert image.dtype == np.uint8
