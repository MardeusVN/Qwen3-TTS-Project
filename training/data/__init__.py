"""
Training data module for Qwen3-TTS training pipeline.

This module provides dataset classes for different training stages:
- Pretraining (S1/S2/S3): Multi-language, long speech support
- DPO: Direct Preference Optimization with chosen/rejected pairs
- GSPO: Group Sparse Policy Optimization with rule-based rewards

Note: For SFT (Speaker Fine-Tuning), use finetuning/dataset.py
"""

from .dataset_pretrain import Qwen3TTSPretrainDataset
from .dataset_dpo import Qwen3TTSDPODataset
from .dataset_gspo import Qwen3TTSGSPODataset

__all__ = [
    "Qwen3TTSPretrainDataset",
    "Qwen3TTSDPODataset",
    "Qwen3TTSGSPODataset",
]