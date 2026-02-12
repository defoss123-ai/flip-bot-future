"""Logging configuration for console and rotating file logs."""

from __future__ import annotations

from pathlib import Path
import sys

from loguru import logger


def setup_logger(log_dir: str | Path = "logs"):
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stdout, level="INFO", enqueue=True)
    logger.add(
        path / "bot.log",
        level="INFO",
        rotation="5 MB",
        retention=5,
        enqueue=True,
        encoding="utf-8",
    )
    return logger
