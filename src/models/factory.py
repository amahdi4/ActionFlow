from __future__ import annotations

from typing import Any

from configs.schema import DataConfig, ModelConfig
from models.base import ClassifierBackend, ModelSummary, PredictionResult


class ResNet18FlowClassifier(ClassifierBackend):
    @classmethod
    def is_available(cls) -> bool:
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
        except ModuleNotFoundError:
            return False
        return True

    @classmethod
    def availability_details(cls) -> str:
        return "Requires PyTorch and torchvision."

    def load(self) -> None:
        raise NotImplementedError("Model loading is scheduled for Phase 2.")

    def predict(self, batch: Any) -> PredictionResult:
        raise NotImplementedError("Model inference is scheduled for Phase 2.")


def build_classifier(model_config: ModelConfig, data_config: DataConfig) -> ClassifierBackend:
    summary = ModelSummary(
        architecture=model_config.architecture,
        num_classes=len(data_config.class_names),
        input_channels=model_config.input_channels,
        pretrained=model_config.pretrained,
        checkpoint=model_config.checkpoint,
        metadata={"class_names": data_config.class_names},
    )

    if model_config.architecture == "resnet18_flow":
        return ResNet18FlowClassifier(summary)

    raise ValueError(f"Unknown classifier architecture: {model_config.architecture}")

