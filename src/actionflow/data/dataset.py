"""Dataset and split helpers for ActionFlow."""

from __future__ import annotations

import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
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

# ---------------------------------------------------------------------------
# KTH sequence annotations
# ---------------------------------------------------------------------------

_SEQUENCE_LINE_RE = re.compile(
    r"^(?P<key>person\d{2}_\w+_d\d)\s+frames\s+(?P<ranges>.+)$"
)


def parse_sequence_annotations(path: str | Path) -> dict[str, list[tuple[int, int]]]:
    """Parse the official KTH ``00sequences.txt`` file.

    Returns a mapping from video key (e.g. ``person01_boxing_d1``) to a list
    of ``(start_frame, end_frame)`` tuples (1-indexed, inclusive, as in the
    annotation file).
    """
    annotations: dict[str, list[tuple[int, int]]] = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        m = _SEQUENCE_LINE_RE.match(line)
        if m is None:
            continue
        key = m.group("key")
        ranges_str = m.group("ranges")
        subsequences: list[tuple[int, int]] = []
        for span in ranges_str.split(","):
            span = span.strip()
            if not span:
                continue
            parts = span.split("-")
            if len(parts) == 2:
                start, end = int(parts[0].strip()), int(parts[1].strip())
                if end >= start:
                    subsequences.append((start, end))
        if subsequences:
            annotations[key] = subsequences
    return annotations


def _dir_name_to_annotation_key(dir_name: str) -> str:
    """Convert a prepared directory name like ``person01_boxing_d1_uncomp`` to annotation key."""
    return dir_name.replace("_uncomp", "")


def build_subsequence_split(
    video_dirs: list[str] | list[Path],
    labels: list[int],
    annotations: dict[str, list[tuple[int, int]]],
) -> tuple[list[str], list[int], list[tuple[int, int]]]:
    """Expand video directories into per-subsequence entries using annotations.

    Returns (dirs, labels, bounds) where bounds[i] = (start, end) in 0-indexed
    frame numbers for the i-th subsequence.  If a video has no annotation entry
    the whole video is kept as a single entry with bounds (0, num_files - 1).
    """
    out_dirs: list[str] = []
    out_labels: list[int] = []
    out_bounds: list[tuple[int, int]] = []
    for video_dir, label in zip(video_dirs, labels):
        vdir = Path(video_dir)
        key = _dir_name_to_annotation_key(vdir.name)
        if key in annotations:
            for start_1, end_1 in annotations[key]:
                # Convert from 1-indexed to 0-indexed
                out_dirs.append(str(vdir))
                out_labels.append(label)
                out_bounds.append((start_1 - 1, end_1 - 1))
        else:
            # Fallback: use the whole video
            out_dirs.append(str(vdir))
            out_labels.append(label)
            out_bounds.append((-1, -1))  # sentinel: use full video
    return out_dirs, out_labels, out_bounds


def filter_split_by_dir_name(
    video_dirs: list[str] | list[Path],
    labels: list[int],
    allowed_names: set[str] | list[str] | tuple[str, ...],
) -> tuple[list[str], list[int]]:
    """Filter a prepared split down to a specific set of video directory names."""
    allowed = set(allowed_names)
    filtered_dirs: list[str] = []
    filtered_labels: list[int] = []
    for video_dir, label in zip(video_dirs, labels):
        if Path(video_dir).name in allowed:
            filtered_dirs.append(str(video_dir))
            filtered_labels.append(label)
    return filtered_dirs, filtered_labels


# ---------------------------------------------------------------------------
# Person ID and split utilities
# ---------------------------------------------------------------------------


def extract_person_id(name: str) -> int:
    """Extract the official KTH person id from a filename or directory name."""
    match = PERSON_PATTERN.search(name)
    if match is None:
        raise ValueError(f"Could not parse KTH person id from {name!r}")
    return int(match.group("id"))


def get_train_val_test_split(
    data_root: str | Path, mode: str = "flow"
) -> tuple[list[str], list[int], list[str], list[int], list[str], list[int]]:
    """Return person-based KTH train/val/test splits for prepared frame or flow directories."""
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


# ---------------------------------------------------------------------------
# Clip index selection
# ---------------------------------------------------------------------------


def select_clip_indices(
    length: int,
    clip_length: int,
    frame_stride: int,
    train: bool,
    *,
    bounds: tuple[int, int] | None = None,
    stride_jitter: bool = False,
    sample_kind: str = "frame",
) -> list[int]:
    """Select temporal indices for a clip within optional subsequence bounds.

    Parameters
    ----------
    length : int
        Total number of files in the video directory.
    clip_length : int
        Number of frames to sample.
    frame_stride : int
        Base stride between sampled frames.
    train : bool
        Random start for training, center for evaluation.
    bounds : tuple[int, int] | None
        Optional (start, end) 0-indexed bounds for the valid subsequence.
        If None or (-1, -1), the full range [0, length) is used.
    stride_jitter : bool
        If True and train, randomly jitter stride by +/- 1.
    """
    range_start, range_end = _resolve_index_range(length, bounds, sample_kind=sample_kind)

    # Determine effective stride
    effective_stride = frame_stride
    if stride_jitter and train:
        effective_stride = random.choice([max(1, frame_stride - 1), frame_stride, frame_stride + 1])

    range_len = range_end - range_start + 1

    required_span = 1 + (clip_length - 1) * effective_stride
    if range_len >= required_span:
        max_start = range_len - required_span
        start_offset = random.randint(0, max_start) if train else max_start // 2
        start = range_start + start_offset
        return [start + offset * effective_stride for offset in range(clip_length)]

    # Fallback: clamp indices within the valid range
    return [min(range_start + offset * effective_stride, range_end) for offset in range(clip_length)]


def select_multi_clip_indices(
    length: int,
    clip_length: int,
    frame_stride: int,
    num_clips: int,
    *,
    bounds: tuple[int, int] | None = None,
    sample_kind: str = "frame",
) -> list[list[int]]:
    """Select evenly-spaced clip start positions for multi-clip evaluation."""
    range_start, range_end = _resolve_index_range(length, bounds, sample_kind=sample_kind)

    range_len = range_end - range_start + 1
    required_span = 1 + (clip_length - 1) * frame_stride

    clips: list[list[int]] = []
    if range_len >= required_span:
        max_start = range_len - required_span
        if num_clips == 1:
            starts = [max_start // 2]
        else:
            starts = [round(i * max_start / (num_clips - 1)) for i in range(num_clips)]
        for s in starts:
            start = range_start + s
            clips.append([start + offset * frame_stride for offset in range(clip_length)])
    else:
        # Not enough frames; just produce one clip
        clip = [min(range_start + offset * frame_stride, range_end) for offset in range(clip_length)]
        clips = [clip] * num_clips

    return clips


def _resolve_index_range(
    length: int,
    bounds: tuple[int, int] | None,
    *,
    sample_kind: str,
) -> tuple[int, int]:
    """Resolve valid inclusive sampling bounds for frames or flow files."""
    if length <= 0:
        raise ValueError("Prepared video directories must contain at least one element.")
    if sample_kind not in {"frame", "flow"}:
        raise ValueError(f"Unsupported sample_kind: {sample_kind!r}")

    if bounds is None or bounds == (-1, -1):
        return 0, length - 1

    range_start = min(max(0, bounds[0]), length - 1)
    raw_end = bounds[1] - 1 if sample_kind == "flow" else bounds[1]
    range_end = min(length - 1, raw_end)
    if range_end < range_start:
        range_end = range_start
    return range_start, range_end


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------


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
        subsequence_bounds: list[tuple[int, int]] | None = None,
    ) -> None:
        self.video_dirs = [Path(path) for path in video_dirs]
        self.labels = labels
        self.resize = tuple(config.resize)
        self.clip_length = int(config.clip_length)
        self.frame_stride = int(config.frame_stride)
        self.input_channels = self.clip_length * 2
        self.train = train
        self.synthetic = synthetic
        self.synthetic_samples = synthetic_samples
        self.subsequence_bounds = subsequence_bounds

    def __len__(self) -> int:
        return self.synthetic_samples if self.synthetic else len(self.video_dirs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        label = self.labels[index % len(self.labels)]
        if self.synthetic:
            sample = torch.randn(self.input_channels, *self.resize, dtype=torch.float32)
            return sample, label

        flow_paths = sorted(self.video_dirs[index].glob("flow_*.npy"))
        bounds = self.subsequence_bounds[index] if self.subsequence_bounds is not None else None
        clip_indices = select_clip_indices(
            len(flow_paths), self.clip_length, self.frame_stride, self.train,
            bounds=bounds, stride_jitter=self.train, sample_kind="flow",
        )
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
        subsequence_bounds: list[tuple[int, int]] | None = None,
    ) -> None:
        self.video_dirs = [Path(path) for path in video_dirs]
        self.labels = labels
        self.resize = tuple(config.resize)
        self.clip_length = int(config.clip_length)
        self.frame_stride = int(config.frame_stride)
        self.train = train
        self.synthetic = synthetic
        self.synthetic_samples = synthetic_samples
        self.subsequence_bounds = subsequence_bounds

    def __len__(self) -> int:
        return self.synthetic_samples if self.synthetic else len(self.video_dirs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        label = self.labels[index % len(self.labels)]
        if self.synthetic:
            sample = torch.randn(3, *self.resize, dtype=torch.float32)
            return sample, label

        frame_paths = sorted(self.video_dirs[index].glob("frame_*.png"))
        bounds = self.subsequence_bounds[index] if self.subsequence_bounds is not None else None
        clip_indices = select_clip_indices(
            len(frame_paths), self.clip_length, self.frame_stride, self.train,
            bounds=bounds, sample_kind="frame",
        )
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
        subsequence_bounds: list[tuple[int, int]] | None = None,
    ) -> None:
        self.video_dirs = [Path(path) for path in video_dirs]
        self.labels = labels
        self.resize = tuple(config.resize)
        self.clip_length = int(config.clip_length)
        self.frame_stride = int(config.frame_stride)
        self.train = train
        self.synthetic = synthetic
        self.synthetic_samples = synthetic_samples
        self.subsequence_bounds = subsequence_bounds

    def __len__(self) -> int:
        return self.synthetic_samples if self.synthetic else len(self.video_dirs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        label = self.labels[index % len(self.labels)]
        if self.synthetic:
            sample = torch.randn(self.clip_length, *self.resize, dtype=torch.float32)
            return sample, label

        frame_paths = sorted(self.video_dirs[index].glob("frame_*.png"))
        bounds = self.subsequence_bounds[index] if self.subsequence_bounds is not None else None
        clip_indices = select_clip_indices(
            len(frame_paths), self.clip_length, self.frame_stride, self.train,
            bounds=bounds, stride_jitter=self.train, sample_kind="frame",
        )

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


# ---------------------------------------------------------------------------
# Multi-clip evaluation
# ---------------------------------------------------------------------------


def multi_clip_evaluate(
    model: nn.Module,
    video_dirs: list[str] | list[Path],
    labels: list[int],
    config: ActionFlowConfig,
    device: str,
    num_clips: int = 10,
    mode: str = "flow",
    subsequence_bounds: list[tuple[int, int]] | None = None,
) -> tuple[list[int], list[int], list[np.ndarray]]:
    """Evaluate by sampling num_clips clips per video and averaging softmax scores.

    Returns (predictions, true_labels, all_avg_probs).
    """
    model.eval()
    resize = tuple(config.resize)
    clip_length = int(config.clip_length)
    frame_stride = int(config.frame_stride)
    target_h, target_w = resize

    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for i, (vdir, label) in enumerate(zip(video_dirs, labels)):
            vdir = Path(vdir)
            bounds = subsequence_bounds[i] if subsequence_bounds is not None else None

            if mode == "flow":
                file_paths = sorted(vdir.glob("flow_*.npy"))
            else:
                file_paths = sorted(vdir.glob("frame_*.png"))

            multi_indices = select_multi_clip_indices(
                len(file_paths),
                clip_length,
                frame_stride,
                num_clips,
                bounds=bounds,
                sample_kind="flow" if mode == "flow" else "frame",
            )

            clip_logits: list[torch.Tensor] = []
            for clip_indices in multi_indices:
                if mode == "flow":
                    flows = [np.load(file_paths[j]).astype(np.float32, copy=False) for j in clip_indices]
                    flow_clip = np.stack(flows, axis=0)
                    flow_clip = np.clip(flow_clip / FLOW_SCALE, -1.0, 1.0)
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
                    input_channels = clip_length * 2
                    tensor = torch.from_numpy(
                        flow_clip.transpose(0, 3, 1, 2).reshape(input_channels, target_h, target_w)
                    ).float().unsqueeze(0).to(device)
                elif mode == "appearance_single":
                    center_idx = clip_indices[len(clip_indices) // 2]
                    frame = cv2.imread(str(file_paths[center_idx]), cv2.IMREAD_COLOR)
                    if frame is None:
                        raise FileNotFoundError(file_paths[center_idx])
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if frame.shape[0] != target_h or frame.shape[1] != target_w:
                        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    t = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0
                    t = (t - IMAGENET_MEAN) / IMAGENET_STD
                    tensor = t.unsqueeze(0).to(device)
                else:  # appearance_temporal
                    frames_list: list[np.ndarray] = []
                    for j in clip_indices:
                        fr = cv2.imread(str(file_paths[j]), cv2.IMREAD_GRAYSCALE)
                        if fr is None:
                            raise FileNotFoundError(file_paths[j])
                        if fr.shape[0] != target_h or fr.shape[1] != target_w:
                            fr = cv2.resize(fr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                        frames_list.append(fr.astype(np.float32, copy=False))
                    frame_clip = np.stack(frames_list, axis=0) / 255.0
                    t = torch.from_numpy(frame_clip).float()
                    t = (t - GRAY_MEAN) / GRAY_STD
                    tensor = t.unsqueeze(0).to(device)

                logits = model(tensor)
                clip_logits.append(logits)

            # Average softmax scores across clips
            all_softmax = torch.cat(clip_logits, dim=0).softmax(dim=1)
            avg_probs = all_softmax.mean(dim=0)
            pred = avg_probs.argmax().item()

            all_preds.append(pred)
            all_labels.append(label)
            all_probs.append(avg_probs.cpu().numpy())

    return all_preds, all_labels, all_probs
