"""Optical-flow helpers for ActionFlow."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def compute_flow(frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray:
    """Compute dense Farneback optical flow between two frames."""
    gray_a = frame_a if frame_a.ndim == 2 else cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = frame_b if frame_b.ndim == 2 else cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray_a,
        gray_b,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    return flow.astype(np.float32, copy=False)


def visualize_flow(flow_array: np.ndarray) -> np.ndarray:
    """Render a dense flow field as an RGB HSV visualization."""
    magnitude, angle = cv2.cartToPolar(flow_array[..., 0], flow_array[..., 1], angleInDegrees=False)
    hsv = np.zeros((*flow_array.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = np.mod(angle * 90 / np.pi, 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def compute_video_flow(frames_dir: str | Path, flow_dir: str | Path) -> dict[str, int | bool]:
    """Cache dense optical flow for every consecutive frame pair in one video directory."""
    frames_path = Path(frames_dir)
    flow_path = Path(flow_dir)
    frame_paths = sorted(frames_path.glob("frame_*.png"))
    if len(frame_paths) < 2:
        return {"expected": 0, "created": 0, "skipped": True}

    flow_path.mkdir(parents=True, exist_ok=True)
    expected = len(frame_paths) - 1
    existing = sorted(flow_path.glob("flow_*.npy"))
    if len(existing) >= expected:
        return {"expected": expected, "created": 0, "skipped": True}

    created = 0
    for index, (first, second) in enumerate(zip(frame_paths[:-1], frame_paths[1:], strict=True)):
        output_path = flow_path / f"flow_{index:05d}.npy"
        if output_path.exists():
            continue
        frame_a = cv2.imread(str(first), cv2.IMREAD_COLOR)
        frame_b = cv2.imread(str(second), cv2.IMREAD_COLOR)
        if frame_a is None or frame_b is None:
            raise FileNotFoundError(f"Could not read frame pair: {first} / {second}")
        np.save(output_path, compute_flow(frame_a, frame_b))
        created += 1

    return {"expected": expected, "created": created, "skipped": created == 0}


def compute_all_flow(data_root: str | Path) -> int:
    """Compute cache-aware dense optical flow under ``data_root/flow`` from extracted frames."""
    root = Path(data_root)
    frames_root = root / "frames"
    flow_root = root / "flow"
    total_created = 0

    if not frames_root.exists():
        return 0

    for class_dir in sorted(path for path in frames_root.iterdir() if path.is_dir()):
        for video_dir in sorted(path for path in class_dir.iterdir() if path.is_dir()):
            relative = video_dir.relative_to(frames_root)
            stats = compute_video_flow(video_dir, flow_root / relative)
            total_created += int(stats["created"])

    return total_created


def summarize_flow_directories(flow_root: str | Path, class_names: tuple[str, ...]) -> list[dict[str, int | str]]:
    """Summarize cached flow availability per action class."""
    root = Path(flow_root)
    rows: list[dict[str, int | str]] = []
    for class_name in class_names:
        class_root = root / class_name
        video_dirs = sorted(path for path in class_root.iterdir() if path.is_dir()) if class_root.exists() else []
        flow_files = sum(len(list(video_dir.glob("flow_*.npy"))) for video_dir in video_dirs)
        rows.append({"class": class_name, "prepared_videos": len(video_dirs), "flow_files": flow_files})
    return rows
