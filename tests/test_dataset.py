"""Tests for ActionFlow datasets."""

from actionflow.config import ActionFlowConfig
from actionflow.data.dataset import FlowClipDataset, RGBClipDataset, TemporalAppearanceClipDataset


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
    config = ActionFlowConfig(mode="rgb")
    dataset = TemporalAppearanceClipDataset([], [0, 1, 2], config=config, synthetic=True, synthetic_samples=4)

    sample, label = dataset[0]

    assert sample.shape == (10, 224, 224)
    assert label in {0, 1, 2}
