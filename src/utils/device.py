from __future__ import annotations

from dataclasses import dataclass
import platform
from types import ModuleType


@dataclass(slots=True)
class DeviceInfo:
    requested: str
    selected: str
    accelerator: str
    mixed_precision: bool
    reason: str
    available_devices: list[str]
    platform: str
    machine: str
    torch_version: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "selected": self.selected,
            "accelerator": self.accelerator,
            "mixed_precision": self.mixed_precision,
            "reason": self.reason,
            "available_devices": self.available_devices,
            "platform": self.platform,
            "machine": self.machine,
            "torch_version": self.torch_version,
        }


def detect_device(preferred: str = "auto") -> DeviceInfo:
    requested = preferred.lower()
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("Device preference must be one of: auto, cuda, mps, cpu")

    system = platform.system().lower()
    machine = platform.machine().lower()
    torch = _import_torch()
    if torch is None:
        return DeviceInfo(
            requested=requested,
            selected="cpu",
            accelerator="cpu",
            mixed_precision=False,
            reason="PyTorch is not installed; falling back to CPU.",
            available_devices=["cpu"],
            platform=system,
            machine=machine,
            torch_version=None,
        )

    availability = {
        "cuda": _cuda_is_available(torch),
        "mps": _mps_is_available(torch),
        "cpu": True,
    }
    available_devices = [name for name in _device_priority(system) if availability[name]]
    torch_version = str(getattr(torch, "__version__", "unknown"))

    if requested == "auto":
        for candidate in _device_priority(system):
            if availability[candidate]:
                return _build_device_info(
                    selected=candidate,
                    requested=requested,
                    reason=_auto_selection_reason(candidate, system),
                    available_devices=available_devices,
                    platform_name=system,
                    machine=machine,
                    torch_version=torch_version,
                )
        return _build_device_info(
            selected="cpu",
            requested=requested,
            reason="No supported accelerator is available; using CPU fallback.",
            available_devices=available_devices,
            platform_name=system,
            machine=machine,
            torch_version=torch_version,
        )

    if availability[requested]:
        return _build_device_info(
            selected=requested,
            requested=requested,
            reason=_requested_selection_reason(requested, system),
            available_devices=available_devices,
            platform_name=system,
            machine=machine,
            torch_version=torch_version,
        )

    fallback = next(candidate for candidate in _device_priority(system) if availability[candidate])
    return _build_device_info(
        selected=fallback,
        requested=requested,
        reason=f"Requested device `{requested}` is unavailable on this runtime; using `{fallback}` instead.",
        available_devices=available_devices,
        platform_name=system,
        machine=machine,
        torch_version=torch_version,
    )


def describe_device(device_info: DeviceInfo) -> str:
    return (
        f"requested={device_info.requested}, "
        f"selected={device_info.selected}, "
        f"accelerator={device_info.accelerator}, "
        f"mixed_precision={device_info.mixed_precision}, "
        f"available_devices={device_info.available_devices}, "
        f"platform={device_info.platform}, "
        f"machine={device_info.machine}, "
        f"torch_version={device_info.torch_version}, "
        f"reason={device_info.reason}"
    )


def _import_torch() -> ModuleType | None:
    try:
        import torch
    except ModuleNotFoundError:
        return None
    return torch


def _device_priority(system: str) -> tuple[str, ...]:
    if system == "darwin":
        return ("mps", "cuda", "cpu")
    return ("cuda", "mps", "cpu")


def _cuda_is_available(torch: ModuleType) -> bool:
    return bool(getattr(torch, "cuda", None) and torch.cuda.is_available())


def _mps_is_available(torch: ModuleType) -> bool:
    backends = getattr(torch, "backends", None)
    mps_backend = getattr(backends, "mps", None)
    return bool(mps_backend and mps_backend.is_available())


def _auto_selection_reason(selected: str, system: str) -> str:
    if selected == "cuda":
        return "CUDA is available and selected for GPU-first execution."
    if selected == "mps":
        return "Apple Metal Performance Shaders is available and selected."
    if system == "darwin":
        return "MPS is unavailable on this Mac runtime; using CPU fallback."
    return "No GPU accelerator is available; using CPU fallback."


def _requested_selection_reason(selected: str, system: str) -> str:
    if selected == "cuda":
        return "CUDA was explicitly requested and is available."
    if selected == "mps":
        return "MPS was explicitly requested and is available."
    if system == "darwin":
        return "CPU was explicitly requested on macOS."
    return "CPU was explicitly requested."


def _build_device_info(
    selected: str,
    requested: str,
    reason: str,
    available_devices: list[str],
    platform_name: str,
    machine: str,
    torch_version: str | None,
) -> DeviceInfo:
    return DeviceInfo(
        requested=requested,
        selected=selected,
        accelerator=selected,
        mixed_precision=selected == "cuda",
        reason=reason,
        available_devices=available_devices,
        platform=platform_name,
        machine=machine,
        torch_version=torch_version,
    )
