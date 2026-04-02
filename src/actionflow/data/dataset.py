"""Dataset and split helpers for ActionFlow."""

from __future__ import annotations

import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from actionflow.config import ActionFlowConfig

PERSON_PATTERN = re.compile(r"person(?P<id>\d{2})_")
TRAIN_PERSONS = set(range(1, 13))
VAL_PERSONS = set(range(13, 17))
TEST_PERSONS = set(range(17, 26))
OFFICIAL_TRAIN_PERSONS = TRAIN_PERSONS | VAL_PERSONS
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
FLOW_SCALE = 20.0
GRAY_MEAN = 0.5
GRAY_STD = 0.25


def extract_person_id(name: str) -> int:
    """Extract the official KTH person id from a filename or directory name."""
    match = PERSON_PATTERN.search(name)
    if match is None:
        raise ValueError(f"Could not parse KTH person id from {name!r}")
    return int(match.group("id"))


def get_train_val_test_split(
    data_root: str | Path, mode: str = "flow"
) -> tuple[list[str], list[int], list[str], list[int], list[str], list[int]]:
    """Return person-based KTH train/val/test splits for prepared frame or flow directories.

    Persons 1-12 train, 13-16 validation (model selection), and 17-25 test (final evaluation).
    """
    root = Path(data_root)
    prepared_root = root / ("flow" if mode == "flow" else "frames")

    class_names = ("boxing", "handclapping", "handwaving", "jogging", "running", "walking")
    train_dirs: list[str] = []
    train_labels: list[int] = []
    val_dirs: list[str] = []
    val_labels: list[int] = []
    test_dirs: list[str] = []
    test_labels: list[int] = []

    for label, class_name in enumerate(class_names):
        class_root = prepared_root / class_name
        if not class_root.exists():
            continue
        class_dirs = sorted(path for path in class_root.iterdir() if path.is_dir())
        for video_dir in class_dirs:
            person_id = extract_person_id(video_dir.name)
            if person_id in TRAIN_PERSONS:
                train_dirs.append(str(video_dir))
                train_labels.append(label)
            elif person_id in VAL_PERSONS:
                val_dirs.append(str(video_dir))
                val_labels.append(label)
            elif person_id in TEST_PERSONS:
                test_dirs.append(str(video_dir))
                test_labels.append(label)

    return train_dirs, train_labels, val_dirs, val_labels, test_dirs, test_labels


def summarize_prepared_split(
    video_dirs: list[str] | list[Path],
    labels: list[int],
    class_names: tuple[str, ...],
) -> list[dict[str, int | str]]:
    """Summarize counts per class for a prepared split."""
    counts = {class_name: 0 for class_name in class_names}
    for label in labels:
        counts[class_names[label]] += 1
    return [{"class": class_name, "videos": counts[class_name]} for class_name in class_names]


def select_clip_indices(length: int, clip_length: int, frame_stride: int, train: bool) -> list[int]:
    """Select temporal indices for a clip, center-biased for eval and random for training."""
    if length <= 0:
        raise ValueError("Prepared video directories must contain at least one element.")

    required_span = 1 + (clip_length - 1) * frame_stride
    if length >= required_span:
        max_start = length - required_span
        start = random.randint(0, max_start) if train else max_start // 2
        return [start + offset * frame_stride for offset in range(clip_length)]
    return [min(offset * frame_stride, length - 1) for offset in range(clip_length)]


class FlowClipDataset(Dataset[tuple[torch.Tensor, int]]):
    """Dataset of stacked dense optical-flow clips."""

    def __init__(
        self,
        video_dirs: list[str] | list[Path],
        labels: list[int],
        config: ActionFlowConfig,
        train: bool = False,
        synthetic: bool = False,
        synthetic_samples: int = 0,
    ) -> None:
        self.video_dirs = [Path(path) for path in video_dirs]
        self.labels = labels
        # Snapshot the shape/sampling config so later notebook mutations do not
        # retroactively change how already-built datasets decode samples.
        self.resize = tuple(config.resize)
        self.clip_length = int(config.clip_length)
        self.frame_stride = int(config.frame_stride)
        self.input_channels = self.clip_length * 2
        self.train = train
        self.synthetic = synthetic
        self.synthetic_samples = synthetic_samples

    def __len__(self) -> int:
        return self.synthetic_samples if self.synthetic else len(self.video_dirs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        label = self.labels[index % len(self.labels)]
        if self.synthetic:
            sample = torch.randn(self.input_channels, *self.resize, dtype=torch.float32)
            return sample, label

        flow_paths = sorted(self.video_dirs[index].glob("flow_*.npy"))
        clip_indices = select_clip_indices(len(flow_paths), self.clip_length, self.frame_stride, self.train)
        flows = [np.load(flow_paths[position]).astype(np.float32, copy=False) for position in clip_indices]
        flow_clip = np.stack(flows, axis=0)
        if self.train and random.random() < 0.5:
            flow_clip = np.flip(flow_clip, axis=2).copy()
            flow_clip[..., 0] *= -1.0

        flow_clip = np.clip(flow_clip / FLOW_SCALE, -1.0, 1.0)
        target_h, target_w = self.resize
        clip_h, clip_w = flow_clip.shape[1], flow_clip.shape[2]
        if clip_h != target_h or clip_w != target_w:
            scale_y, scale_x = target_h / clip_h, target_w / clip_w
            resized = np.stack(
                [cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_LINEAR) for f in flow_clip],
                axis=0,
            )
            resized[..., 0] *= scale_x
            resized[..., 1] *= scale_y
            flow_clip = resized
        tensor = torch.from_numpy(flow_clip.transpose(0, 3, 1, 2).reshape(self.input_channels, *self.resize))
        return tensor.float(), label


class RGBClipDataset(Dataset[tuple[torch.Tensor, int]]):
    """Dataset of center RGB frames sampled from KTH clips."""

    def __init__(
        self,
        video_dirs: list[str] | list[Path],
        labels: list[int],
        config: ActionFlowConfig,
        train: bool = False,
        synthetic: bool = False,
        synthetic_samples: int = 0,
    ) -> None:
        self.video_dirs = [Path(path) for path in video_dirs]
        self.labels = labels
        self.resize = tuple(config.resize)
        self.clip_length = int(config.clip_length)
        self.frame_stride = int(config.frame_stride)
        self.train = train
        self.synthetic = synthetic
        self.synthetic_samples = synthetic_samples

    def __len__(self) -> int:
        return self.synthetic_samples if self.synthetic else len(self.video_dirs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        label = self.labels[index % len(self.labels)]
        if self.synthetic:
            sample = torch.randn(3, *self.resize, dtype=torch.float32)
            return sample, label

        frame_paths = sorted(self.video_dirs[index].glob("frame_*.png"))
        clip_indices = select_clip_indices(len(frame_paths), self.clip_length, self.frame_stride, self.train)
        center_index = clip_indices[len(clip_indices) // 2]
        frame = cv2.imread(str(frame_paths[center_index]), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(frame_paths[center_index])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        target_h, target_w = self.resize
        if frame.shape[0] != target_h or frame.shape[1] != target_w:
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        if self.train and random.random() < 0.5:
            frame = np.flip(frame, axis=1).copy()
        tensor = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        return tensor, label


class TemporalAppearanceClipDataset(Dataset[tuple[torch.Tensor, int]]):
    """Dataset of stacked grayscale appearance frames sampled over time."""

    def __init__(
        self,
        video_dirs: list[str] | list[Path],
        labels: list[int],
        config: ActionFlowConfig,
        train: bool = False,
        synthetic: bool = False,
        synthetic_samples: int = 0,
    ) -> None:
        self.video_dirs = [Path(path) for path in video_dirs]
        self.labels = labels
        self.resize = tuple(config.resize)
        self.clip_length = int(config.clip_length)
        self.frame_stride = int(config.frame_stride)
        self.train = train
        self.synthetic = synthetic
        self.synthetic_samples = synthetic_samples

    def __len__(self) -> int:
        return self.synthetic_samples if self.synthetic else len(self.video_dirs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        label = self.labels[index % len(self.labels)]
        if self.synthetic:
            sample = torch.randn(self.clip_length, *self.resize, dtype=torch.float32)
            return sample, label

        frame_paths = sorted(self.video_dirs[index].glob("frame_*.png"))
        clip_indices = select_clip_indices(len(frame_paths), self.clip_length, self.frame_stride, self.train)

        target_h, target_w = self.resize
        frames: list[np.ndarray] = []
        for position in clip_indices:
            frame = cv2.imread(str(frame_paths[position]), cv2.IMREAD_GRAYSCALE)
            if frame is None:
                raise FileNotFoundError(frame_paths[position])
            if frame.shape[0] != target_h or frame.shape[1] != target_w:
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            frames.append(frame.astype(np.float32, copy=False))

        frame_clip = np.stack(frames, axis=0) / 255.0
        if self.train and random.random() < 0.5:
            frame_clip = np.flip(frame_clip, axis=2).copy()

        tensor = torch.from_numpy(frame_clip)
        tensor = (tensor - GRAY_MEAN) / GRAY_STD
        return tensor.float(), label
