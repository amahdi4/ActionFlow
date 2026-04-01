from __future__ import annotations

import logging

from configs.schema import AppConfig
from data.datasets import summarize_dataset_config, validate_dataset_name
from flow import build_flow_estimator
from models import build_classifier
from representations import build_representation
from utils.device import describe_device, detect_device
from utils.runtime import ensure_directory, get_project_root

logger = logging.getLogger(__name__)


class TrainingPipeline:
    def __init__(self, config: AppConfig) -> None:
        validate_dataset_name(config.data.dataset_name)
        self.config = config
        self.project_root = get_project_root()
        self.device_info = detect_device(config.runtime.device)
        self.flow_estimator = build_flow_estimator(config.flow.backend, params=config.flow.params)
        self.representation = build_representation(config.flow.representation)
        self.classifier = build_classifier(config.model, config.data)
        ensure_directory(self.project_root / config.project.output_dir)
        logger.info(
            "Initialized pipeline | dataset=%s | requested_device=%s | selected_device=%s | flow=%s | representation=%s | model=%s",
            config.data.dataset_name,
            config.runtime.device,
            self.device_info.selected,
            config.flow.backend,
            config.flow.representation,
            config.model.architecture,
        )

    def dry_run_summary(self) -> dict[str, object]:
        dataset_summary = summarize_dataset_config(self.config.data)
        logger.info(
            "Building dry-run summary | selected_device=%s | accelerator=%s | flow_available=%s | model_available=%s",
            self.device_info.selected,
            self.device_info.accelerator,
            self.flow_estimator.summary()["available"],
            self.classifier.summary()["available"],
        )
        return {
            "project": {
                "name": self.config.project.name,
                "project_root": str(self.project_root),
                "output_dir": str(self.project_root / self.config.project.output_dir),
            },
            "runtime": {
                "requested_device": self.device_info.requested,
                "device": self.device_info.selected,
                "accelerator": self.device_info.accelerator,
                "mixed_precision": self.device_info.mixed_precision,
                "available_devices": self.device_info.available_devices,
                "platform": self.device_info.platform,
                "machine": self.device_info.machine,
                "torch_version": self.device_info.torch_version,
                "details": describe_device(self.device_info),
            },
            "dataset": dataset_summary,
            "flow": self.flow_estimator.summary(),
            "representation": self.representation.summary(),
            "model": self.classifier.summary(),
            "notes": [
                "Phase 1 scaffold only. No optical flow computation or model inference is executed yet.",
                "The dry run validates configuration wiring and runtime selection.",
            ],
        }
