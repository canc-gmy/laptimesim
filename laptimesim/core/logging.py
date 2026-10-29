"""Structured logging configuration for laptimesim."""

import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Configure structured logging for the application.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file: Optional path to write log output.

    Returns:
        Configured root logger for laptimesim.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger("laptimesim")
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if setup_logging is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric_level)
        logger.addHandler(file_handler)

    return logger
