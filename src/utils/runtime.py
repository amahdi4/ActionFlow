from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(raw_path: str | Path, base: str | Path | None = None) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    base_path = Path(base).expanduser().resolve() if base is not None else get_project_root()
    return (base_path / candidate).resolve()


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory

