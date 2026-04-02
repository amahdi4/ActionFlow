"""Command-line interface for ActionFlow."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from actionflow.config import ActionFlowConfig
from actionflow.data import (
    FlowClipDataset,
    RGBClipDataset,
    compute_all_flow,
    extract_all_frames,
    get_train_val_test_split,
)
from actionflow.evaluation import evaluate_model
from actionflow.models import build_resnet18_flow
from actionflow.training import Trainer
from actionflow.training.metrics import plot_training_curves
from actionflow.utils.device import resolve_device
from actionflow.utils.seed import seed_everything


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level ActionFlow CLI parser."""
    parser = argparse.ArgumentParser(prog="actionflow", description="Optical flow action recognition on KTH.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke_parser = subparsers.add_parser("smoke-test", help="Run the synthetic CPU smoke test.")
    smoke_parser.add_argument("--output-dir", default="outputs/smoke", help="Directory for smoke-test artifacts.")

    prepare_parser = subparsers.add_parser("prepare-data", help="Extract frames and optical flow for KTH.")
    _add_common_data_args(prepare_parser)

    train_parser = subparsers.add_parser("train", help="Train a flow or RGB classifier.")
    _add_common_runtime_args(train_parser)
    train_parser.add_argument("--mode", choices=("flow", "rgb"), default="flow")
    train_parser.add_argument("--epochs", type=int, default=25)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--subset", type=int, default=None)
    train_parser.add_argument("--clip-length", type=int, default=10)
    train_parser.add_argument("--frame-stride", type=int, default=2)
    train_parser.add_argument("--pretrained-backbone", action=argparse.BooleanOptionalAction, default=True)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a trained checkpoint on the KTH test split.")
    _add_common_runtime_args(evaluate_parser)
    evaluate_parser.add_argument("--mode", choices=("flow", "rgb"), default="flow")
    evaluate_parser.add_argument("--batch-size", type=int, default=16)
    evaluate_parser.add_argument("--clip-length", type=int, default=10)
    evaluate_parser.add_argument("--frame-stride", type=int, default=2)
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--pretrained-backbone", action=argparse.BooleanOptionalAction, default=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ActionFlow CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "smoke-test":
        run_smoke_test(output_dir=args.output_dir)
        return 0
    if args.command == "prepare-data":
        config = _config_from_args(args)
        prepare_data_command(config)
        return 0
    if args.command == "train":
        config = _config_from_args(args)
        train_command(config)
        return 0
    if args.command == "evaluate":
        config = _config_from_args(args)
        evaluate_command(config, checkpoint=args.checkpoint)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


def prepare_data_command(config: ActionFlowConfig) -> None:
    """Extract frames and dense optical flow from raw KTH videos."""
    frames = extract_all_frames(config.data_root, config.resize)
    flows = compute_all_flow(config.data_root)
    print(f"Prepared data: extracted {frames} frames and cached {flows} flow files.")


def train_command(config: ActionFlowConfig) -> dict[str, list[float]]:
    """Train a model on the prepared KTH dataset."""
    seed_everything(config.seed)
    device = resolve_device(config.device)
    train_loader, val_loader = _build_train_val_loaders(config)
    model = build_resnet18_flow(config.num_classes, config.input_channels, pretrained=config.pretrained_backbone)
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, config=config, device=device)
    history = trainer.train()
    figure = plot_training_curves(history, Path(config.output_dir) / f"training_curves_{config.mode}.png")
    plt.close(figure)
    print(f"Saved checkpoint to {Path(config.output_dir) / f'best_{config.mode}.pt'}")
    return history


def evaluate_command(config: ActionFlowConfig, checkpoint: str) -> dict[str, object]:
    """Evaluate a trained checkpoint on the KTH test split."""
    seed_everything(config.seed)
    device = resolve_device(config.device)
    _, test_loader = _build_train_val_loaders(config)
    model = build_resnet18_flow(config.num_classes, config.input_channels, pretrained=config.pretrained_backbone)
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    return evaluate_model(model, test_loader, config.class_names, device, config.output_dir)


def run_smoke_test(output_dir: str = "outputs/smoke") -> dict[str, object]:
    """Run a full synthetic train/eval loop and assert the expected artifacts."""
    start = time.perf_counter()
    config = ActionFlowConfig(
        mode="flow",
        clip_length=4,
        resize=(64, 64),
        num_classes=3,
        class_names=("boxing", "walking", "running"),
        batch_size=4,
        epochs=2,
        lr=1e-2,
        pretrained_backbone=False,
        output_dir=output_dir,
        smoke_test=True,
    )
    seed_everything(config.seed)

    train_labels = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    val_labels = [0, 1, 2, 0]
    train_dataset = FlowClipDataset([], train_labels, config=config, train=True, synthetic=True, synthetic_samples=16)
    val_dataset = FlowClipDataset([], val_labels, config=config, train=False, synthetic=True, synthetic_samples=4)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    model = build_resnet18_flow(
        num_classes=config.num_classes,
        input_channels=config.input_channels,
        pretrained=config.pretrained_backbone,
    )
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, config=config, device=config.device)
    history = trainer.train()
    figure = plot_training_curves(history, Path(output_dir) / "training_curves.png")
    plt.close(figure)
    metrics = evaluate_model(model=trainer.model, test_loader=val_loader, class_names=config.class_names, device="cpu", output_dir=output_dir)

    checkpoint_path = Path(output_dir) / f"best_{config.mode}.pt"
    curves_path = Path(output_dir) / "training_curves.png"
    metrics_path = Path(output_dir) / "metrics.json"
    confusion_path = Path(output_dir) / "confusion_matrix.png"

    assert history["train_loss"][-1] <= history["train_loss"][0], "Smoke test loss did not decrease."
    assert checkpoint_path.exists(), "Smoke test checkpoint was not created."
    assert curves_path.exists(), "Smoke test training-curve plot was not created."
    assert metrics_path.exists(), "Smoke test metrics JSON was not created."
    assert confusion_path.exists(), "Smoke test confusion matrix was not created."

    elapsed = time.perf_counter() - start
    assert elapsed < 60, f"Smoke test exceeded time budget: {elapsed:.2f}s"
    print("SMOKE TEST PASSED")
    return {"history": history, "metrics": metrics, "elapsed_seconds": elapsed}


def _build_train_val_loaders(config: ActionFlowConfig) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders for prepared KTH data."""
    train_dirs, train_labels, val_dirs, val_labels, _test_dirs, _test_labels = get_train_val_test_split(config.data_root, mode=config.mode)
    if config.subset is not None:
        train_dirs, train_labels = _limit_per_class(train_dirs, train_labels, config.subset)
        val_dirs, val_labels = _limit_per_class(val_dirs, val_labels, config.subset)

    dataset_cls = FlowClipDataset if config.mode == "flow" else RGBClipDataset
    train_dataset = dataset_cls(train_dirs, train_labels, config=config, train=True)
    val_dataset = dataset_cls(val_dirs, val_labels, config=config, train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader


def _config_from_args(args: argparse.Namespace) -> ActionFlowConfig:
    """Build an ``ActionFlowConfig`` from CLI arguments."""
    resize = tuple(args.resize)
    return ActionFlowConfig(
        data_root=args.data_root,
        clip_length=getattr(args, "clip_length", 10),
        frame_stride=getattr(args, "frame_stride", 2),
        resize=(int(resize[0]), int(resize[1])),
        mode=getattr(args, "mode", "flow"),
        pretrained_backbone=getattr(args, "pretrained_backbone", True),
        batch_size=getattr(args, "batch_size", 16),
        epochs=getattr(args, "epochs", 25),
        lr=getattr(args, "lr", 1e-3),
        weight_decay=getattr(args, "weight_decay", 1e-4),
        device=getattr(args, "device", "cpu"),
        num_workers=0,
        output_dir=getattr(args, "output_dir", "outputs"),
        subset=getattr(args, "subset", None),
    )


def _limit_per_class(video_dirs: list[str], labels: list[int], subset: int) -> tuple[list[str], list[int]]:
    """Limit prepared video directories to at most ``subset`` items per class."""
    counts: dict[int, int] = {}
    limited_dirs: list[str] = []
    limited_labels: list[int] = []
    for video_dir, label in zip(video_dirs, labels, strict=True):
        current = counts.get(label, 0)
        if current >= subset:
            continue
        counts[label] = current + 1
        limited_dirs.append(video_dir)
        limited_labels.append(label)
    return limited_dirs, limited_labels


def _add_common_data_args(parser: argparse.ArgumentParser) -> None:
    """Add shared data-preparation arguments to a subparser."""
    parser.add_argument("--data-root", default="data/kth")
    parser.add_argument("--resize", nargs=2, type=int, default=(224, 224), metavar=("HEIGHT", "WIDTH"))


def _add_common_runtime_args(parser: argparse.ArgumentParser) -> None:
    """Add shared runtime arguments to a subparser."""
    _add_common_data_args(parser)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs")
