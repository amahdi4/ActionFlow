from __future__ import annotations

import argparse
import json
import sys

from configs.loader import load_config, load_default_config
from demo.app import launch_demo
from training.pipeline import TrainingPipeline
from utils.device import detect_device
from utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ActionFlow command line interface.")
    parser.add_argument("--config", type=str, help="Path to a JSON or YAML config file.")
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=["train", "demo"],
        help="Config profile to use when --config is not provided. Defaults to `demo` for launch and `train` otherwise.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Override the runtime log level from the config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cuda", "mps", "cpu"],
        help="Override the runtime device without editing the config.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check-env", help="Print the resolved runtime device environment.")
    subparsers.add_parser("show-config", help="Print the resolved configuration.")
    subparsers.add_parser("dry-run", help="Print a wiring summary without training or inference.")
    subparsers.add_parser(
        "launch",
        aliases=["launch-demo"],
        help="Launch the Gradio demo scaffold.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    profile = _resolve_profile(args.command, args.profile)
    config = load_config(args.config) if args.config else load_default_config(profile)
    if args.device is not None:
        config.runtime.device = args.device
    if args.log_level is not None:
        config.runtime.log_level = args.log_level

    configure_logging(config.runtime.log_level)

    if args.command == "check-env":
        print(json.dumps(detect_device(config.runtime.device).to_dict(), indent=2))
        return 0

    if args.command == "show-config":
        print(json.dumps(config.to_dict(), indent=2))
        return 0

    if args.command == "dry-run":
        pipeline = TrainingPipeline(config)
        print(json.dumps(pipeline.dry_run_summary(), indent=2))
        return 0

    if args.command in {"launch", "launch-demo"}:
        launch_demo(config)
        return 0

    parser.print_help()
    return 1


def _resolve_profile(command: str, requested_profile: str | None) -> str:
    if requested_profile is not None:
        return requested_profile
    if command in {"launch", "launch-demo"}:
        return "demo"
    return "train"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
