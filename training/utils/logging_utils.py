# training/utils/logging_utils.py
"""
Logging utilities for Qwen3-TTS training.
"""

import logging
import os
import sys


def setup_logger(
    name: str = "qwen3-tts",
    level: int = logging.INFO,
    log_file: str = None,
) -> logging.Logger:
    """
    Setup logger for Qwen3-TTS training.

    Args:
        name: Logger name.
        level: Logging level.
        log_file: Path to log file. If None, logs to stdout only.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler if log_file is specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger