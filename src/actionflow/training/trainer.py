"""Model training loop for ActionFlow."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader

from actionflow.config import ActionFlowConfig
from actionflow.training.metrics import compute_accuracy


class Trainer:
    """End-to-end trainer with validation and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: ActionFlowConfig,
        device: str,
    ) -> None:
        """Initialize trainer state for a given model and dataloaders."""
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = self._build_scheduler()
        self.best_val_acc = float("-inf")
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    def train(self) -> dict[str, list[float]]:
        """Run the training loop and return the recorded history."""
        output_dir = Path(self.config.output_dir)
        checkpoint_path = output_dir / f"best_{self.config.mode}.pt"
        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            if self.scheduler is not None:
                self.scheduler.step()
            if val_acc >= self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, checkpoint_path)

            print(
                f"Epoch {epoch}/{self.config.epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / f"history_{self.config.mode}.json").open("w", encoding="utf-8") as handle:
            json.dump(self.history, handle, indent=2)
        return self.history

    def train_one_epoch(self, epoch: int) -> tuple[float, float]:
        """Train for one epoch and return average loss and accuracy."""
        del epoch
        self.model.train()
        total_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []

        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(inputs)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            all_preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

        average_loss = total_loss / max(len(self.train_loader.dataset), 1)
        accuracy = compute_accuracy(all_preds, all_labels)
        return average_loss, accuracy

    def validate(self) -> tuple[float, float]:
        """Evaluate the current model on the validation loader."""
        self.model.eval()
        total_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)

                total_loss += loss.item() * inputs.size(0)
                all_preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
                all_labels.extend(labels.detach().cpu().tolist())

        average_loss = total_loss / max(len(self.val_loader.dataset), 1)
        accuracy = compute_accuracy(all_preds, all_labels)
        return average_loss, accuracy

    def save_checkpoint(self, epoch: int, path: str | Path) -> None:
        """Save a training checkpoint to disk."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_acc": self.best_val_acc,
                "config": asdict(self.config),
            },
            output_path,
        )

    def _build_scheduler(self) -> LRScheduler | None:
        """Create the configured learning-rate scheduler."""
        if self.config.scheduler == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=max(self.config.epochs, 1))
        return None
