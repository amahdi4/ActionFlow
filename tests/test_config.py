"""Tests for ActionFlow configuration."""

from actionflow.config import ActionFlowConfig


def test_config_defaults() -> None:
    """Default config values should match the project contract."""
    config = ActionFlowConfig()

    assert config.data_root == "data/kth"
    assert config.mode == "flow"
    assert config.clip_length == 10
    assert config.input_channels == 20
    assert config.device == "cpu"
    assert config.num_workers == 0


def test_flow_mode_input_channels_are_derived() -> None:
    """Flow mode should map clip length to twice as many channels."""
    config = ActionFlowConfig(mode="flow", clip_length=12, input_channels=99)

    assert config.input_channels == 24


def test_rgb_mode_input_channels_are_derived() -> None:
    """RGB mode should force three input channels."""
    config = ActionFlowConfig(mode="rgb", input_channels=99)

    assert config.input_channels == 3


def test_class_count_matches_class_names() -> None:
    """The configured number of classes should equal the class-name tuple length."""
    config = ActionFlowConfig()

    assert config.num_classes == len(config.class_names)
