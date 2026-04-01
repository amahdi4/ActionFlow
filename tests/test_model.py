"""Tests for the modified ResNet18 backbone."""

import torch

from actionflow.models.resnet_flow import _replicate_conv1_weights, build_resnet18_flow


def test_flow_model_forward_shape() -> None:
    """Flow inputs should produce class logits with the expected shape."""
    model = build_resnet18_flow(num_classes=6, input_channels=20, pretrained=False)
    batch = torch.randn(2, 20, 224, 224)

    output = model(batch)

    assert output.shape == (2, 6)


def test_rgb_model_forward_shape() -> None:
    """RGB inputs should produce class logits with the expected shape."""
    model = build_resnet18_flow(num_classes=6, input_channels=3, pretrained=False)
    batch = torch.randn(2, 3, 224, 224)

    output = model(batch)

    assert output.shape == (2, 6)


def test_weight_replication_shape_and_scaling() -> None:
    """Replicated conv weights should match the requested channel count."""
    weight = torch.arange(64 * 3 * 7 * 7, dtype=torch.float32).view(64, 3, 7, 7)

    replicated = _replicate_conv1_weights(weight, input_channels=20)

    assert replicated.shape == (64, 20, 7, 7)
    assert torch.allclose(replicated[:, 0], weight[:, 0] * (3 / 20))
    assert torch.allclose(replicated[:, 3], weight[:, 0] * (3 / 20))
