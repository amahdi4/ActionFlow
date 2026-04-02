"""Frame extraction helpers for ActionFlow."""

from __future__ import annotations

from pathlib import Path

import cv2


def extract_video_frames(avi_path: str | Path, output_dir: str | Path, resize: tuple[int, int]) -> dict[str, int | bool]:
    """Extract resized PNG frames for one KTH AVI file with cache-aware skipping."""
    source = Path(avi_path)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {source}")

    total_frames = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
    existing_frames = sorted(target.glob("frame_*.png"))
    if total_frames > 0 and len(existing_frames) >= total_frames:
        capture.release()
        return {"expected": total_frames, "created": 0, "skipped": True}

    start_index = len(existing_frames)
    if start_index > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_index)

    height, width = resize
    frame_index = start_index
    while True:
        success, frame = capture.read()
        if not success:
            break
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(target / f"frame_{frame_index:05d}.png"), resized)
        frame_index += 1

    capture.release()
    return {"expected": frame_index, "created": frame_index - start_index, "skipped": frame_index == start_index}


def extract_all_frames(data_root: str | Path, resize: tuple[int, int]) -> int:
    """Extract cache-aware PNG frames under ``data_root/frames`` from ``data_root/raw``."""
    root = Path(data_root)
    raw_root = root / "raw"
    frames_root = root / "frames"
    total_created = 0

    if not raw_root.exists():
        return 0

    for class_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        for avi_path in sorted(class_dir.glob("*.avi")):
            relative = avi_path.relative_to(raw_root).with_suffix("")
            stats = extract_video_frames(avi_path, frames_root / relative, resize)
            total_created += int(stats["created"])

    return total_created


def summarize_frame_directories(frames_root: str | Path, class_names: tuple[str, ...]) -> list[dict[str, int | str]]:
    """Summarize extracted frame availability per action class."""
    root = Path(frames_root)
    rows: list[dict[str, int | str]] = []
    for class_name in class_names:
        class_root = root / class_name
        video_dirs = sorted(path for path in class_root.iterdir() if path.is_dir()) if class_root.exists() else []
        frame_files = sum(len(list(video_dir.glob("frame_*.png"))) for video_dir in video_dirs)
        rows.append({"class": class_name, "prepared_videos": len(video_dirs), "frame_files": frame_files})
    return rows
