"""Tests for ActionFlow datasets."""

from pathlib import Path

from actionflow.config import ActionFlowConfig
from actionflow.data.dataset import (
    FlowClipDataset,
    RGBClipDataset,
    TemporalAppearanceClipDataset,
    build_subsequence_split,
    parse_sequence_annotations,
    select_clip_indices,
    select_multi_clip_indices,
)


def test_synthetic_flow_dataset_shape_and_label() -> None:
    """Synthetic flow samples should have stacked flow channels."""
    config = ActionFlowConfig(mode="flow")
    dataset = FlowClipDataset([], [0, 1, 2], config=config, synthetic=True, synthetic_samples=4)

    sample, label = dataset[0]

    assert sample.shape == (20, 224, 224)
    assert label in {0, 1, 2}


def test_synthetic_rgb_dataset_shape_and_label() -> None:
    """Synthetic RGB samples should match the ResNet RGB contract."""
    config = ActionFlowConfig(mode="rgb")
    dataset = RGBClipDataset([], [0, 1, 2], config=config, synthetic=True, synthetic_samples=4)

    sample, label = dataset[0]

    assert sample.shape == (3, 224, 224)
    assert label in {0, 1, 2}


def test_synthetic_temporal_appearance_dataset_shape_and_label() -> None:
    """Synthetic temporal appearance samples should stack grayscale frames over time."""
    config = ActionFlowConfig(mode="appearance_temporal")
    dataset = TemporalAppearanceClipDataset([], [0, 1, 2], config=config, synthetic=True, synthetic_samples=4)

    sample, label = dataset[0]

    assert sample.shape == (10, 224, 224)
    assert label in {0, 1, 2}


def test_parse_sequence_annotations_and_expand_subsequences(tmp_path: Path) -> None:
    """Official annotation lines should expand each video into per-subsequence samples."""
    annotation_path = tmp_path / "00sequences.txt"
    annotation_path.write_text(
        "\n".join(
            [
                "person01_boxing_d1\t\tframes\t1-95, 96-185, 186-245, 246-360",
                "person01_running_d1\t\tframes\t1-35, 95-130, 202-235, 295-335",
            ]
        ),
        encoding="utf-8",
    )

    annotations = parse_sequence_annotations(annotation_path)
    video_dirs, labels, bounds = build_subsequence_split(
        ["data/kth/flow/boxing/person01_boxing_d1_uncomp"],
        [0],
        annotations,
    )

    assert annotations["person01_boxing_d1"][0] == (1, 95)
    assert len(video_dirs) == 4
    assert labels == [0, 0, 0, 0]
    assert bounds[0] == (0, 94)
    assert bounds[-1] == (245, 359)


def test_flow_sampling_bounds_exclude_subsequence_end_transition() -> None:
    """Flow clips should stop one frame earlier than RGB clips inside a subsequence."""
    frame_indices = select_clip_indices(
        length=20,
        clip_length=4,
        frame_stride=1,
        train=False,
        bounds=(5, 14),
        sample_kind="frame",
    )
    flow_indices = select_clip_indices(
        length=19,
        clip_length=4,
        frame_stride=1,
        train=False,
        bounds=(5, 14),
        sample_kind="flow",
    )

    assert frame_indices == [8, 9, 10, 11]
    assert flow_indices == [7, 8, 9, 10]
    assert max(flow_indices) == max(frame_indices) - 1


def test_multi_clip_indices_respect_flow_subsequence_bounds() -> None:
    """Multi-clip sampling should stay within the adjusted flow subsequence range."""
    clips = select_multi_clip_indices(
        length=19,
        clip_length=4,
        frame_stride=1,
        num_clips=3,
        bounds=(5, 14),
        sample_kind="flow",
    )

    assert clips[0] == [5, 6, 7, 8]
    assert clips[-1] == [10, 11, 12, 13]
