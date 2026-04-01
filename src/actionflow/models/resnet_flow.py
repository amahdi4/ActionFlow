"""ResNet backbones for ActionFlow."""

from __future__ import annotations

import warnings

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def _replicate_conv1_weights(weight: torch.Tensor, input_channels: int) -> torch.Tensor:
    """Expand pretrained RGB conv weights to an arbitrary input channel count.

    Each new input channel reuses one of the original RGB filters in round-robin
    order and scales by ``3 / input_channels`` so the summed activation energy
    stays close to the pretrained RGB initialization.
    """
    if weight.ndim != 4:
        raise ValueError("conv1 weight must be a 4D tensor.")
    if input_channels <= 0:
        raise ValueError("input_channels must be positive.")

    repeated = torch.stack(
        [weight[:, channel % weight.shape[1], :, :] for channel in range(input_channels)],
        dim=1,
    )
    return repeated * (weight.shape[1] / input_channels)


def build_resnet18_flow(num_classes: int, input_channels: int, pretrained: bool = True) -> nn.Module:
    """Build a ResNet18 classifier for optical-flow or RGB inputs.

    When ``input_channels`` differs from 3 and pretrained weights are enabled,
    the first convolution is inflated from RGB weights using:

    ``W_new[:, c, :, :] = W_rgb[:, c % 3, :, :] * (3 / input_channels)``

    This preserves the expected activation scale while allowing any channel
    count, including stacked optical-flow inputs of shape ``2 * clip_length``.
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    if input_channels <= 0:
        raise ValueError("input_channels must be positive.")

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    try:
        model = resnet18(weights=weights)
    except Exception as exc:  # pragma: no cover - defensive offline fallback
        warnings.warn(
            f"Falling back to randomly initialized ResNet18 because pretrained weights could not be loaded: {exc}",
            stacklevel=2,
        )
        model = resnet18(weights=None)
        weights = None

    old_conv = model.conv1
    new_conv = nn.Conv2d(
        input_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    with torch.no_grad():
        if weights is not None:
            new_conv.weight.copy_(_replicate_conv1_weights(old_conv.weight.data, input_channels))
        elif input_channels == old_conv.in_channels:
            new_conv.weight.copy_(old_conv.weight.data)
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

    model.conv1 = new_conv
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
