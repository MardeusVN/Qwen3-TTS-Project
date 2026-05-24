# training/utils/checkpoint_utils.py
"""
Checkpoint utilities for Qwen3-TTS training.
"""

import os
import torch
from typing import Tuple

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    step: int,
    checkpoint_dir: str,
    filename: str = "checkpoint.pt",
) -> None:
    """
    Save model checkpoint.

    Args:
        model: Model to save.
        optimizer: Optimizer to save.
        scheduler: Learning rate scheduler to save.
        epoch: Current epoch.
        step: Current step.
        checkpoint_dir: Directory to save checkpoint.
        filename: Filename for checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    checkpoint_path: str,
) -> Tuple[int, int]:
    """
    Load model checkpoint.

    Args:
        model: Model to load checkpoint into.
        optimizer: Optimizer to load checkpoint into.
        scheduler: Learning rate scheduler to load checkpoint into.
        checkpoint_path: Path to checkpoint file.

    Returns:
        Tuple[int, int]: Epoch and step from checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint["epoch"]
    step = checkpoint["step"]

    print(f"Checkpoint loaded from {checkpoint_path} (epoch {epoch}, step {step})")

    return epoch, step