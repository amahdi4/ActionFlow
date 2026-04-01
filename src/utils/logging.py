from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    normalized = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=normalized,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
