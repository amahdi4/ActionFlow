from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from configs.schema import AppConfig


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    payload = _read_mapping(config_path)
    return AppConfig.from_mapping(payload)


def load_default_config(profile: str = "train") -> AppConfig:
    defaults_dir = Path(__file__).resolve().parent / "defaults"
    config_path = defaults_dir / f"{profile}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Unknown config profile: {profile}")
    return load_config(config_path)


def dump_config(config: AppConfig, path: str | Path) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PyYAML is required to write YAML configs. Install with `pip install PyYAML`."
            ) from exc
        with destination.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
        return

    with destination.open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            payload = json.load(handle)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "PyYAML is required to read YAML configs. Install with `pip install PyYAML`."
                ) from exc
            payload = yaml.safe_load(handle)
        else:
            raise ValueError(f"Unsupported config format: {suffix}")

    if not isinstance(payload, dict):
        raise TypeError("Configuration payload must be a mapping.")
    return payload

