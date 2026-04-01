"""Tests for the training loop."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from actionflow.config import ActionFlowConfig
from actionflow.data.dataset import FlowClipDataset
from actionflow.models import build_resnet18_flow
from actionflow.training.trainer import Trainer


def test_trainer_runs_one_epoch_and_updates_gradients(tmp_path: Path) -> None:
    """A single training epoch should produce finite loss and non-zero gradients."""
    config = ActionFlowConfig(
        mode="flow",
        clip_length=4,
        resize=(64, 64),
        batch_size=4,
        epochs=1,
        lr=1e-2,
        output_dir=str(tmp_path),
    )
    train_dataset = FlowClipDataset([], [0, 1, 2, 0, 1, 2, 0, 1], config=config, synthetic=True, synthetic_samples=8)
    val_dataset = FlowClipDataset([], [0, 1, 2, 0], config=config, synthetic=True, synthetic_samples=4)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    model = build_resnet18_flow(num_classes=config.num_classes, input_channels=config.input_channels, pretrained=False)

    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, config=config, device="cpu")
    train_loss, train_acc = trainer.train_one_epoch(epoch=1)

    gradients = [parameter.grad for parameter in trainer.model.parameters() if parameter.grad is not None]

    assert torch.isfinite(torch.tensor(train_loss))
    assert 0.0 <= train_acc <= 1.0
    assert gradients
    assert any(torch.any(gradient != 0) for gradient in gradients)
