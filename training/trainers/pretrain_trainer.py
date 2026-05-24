# training/trainers/pretrain_trainer.py
# -*- coding: utf-8 -*-
"""
Pre-training Trainer for Qwen3-TTS (S1/S2/S3 stages).

This trainer implements pre-training with FSDP support for large-scale training.
It wraps HuggingFace Trainer with custom loss computation.
"""
import os
import os
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from qwen_tts.core.models.processing_qwen3_tts import Qwen3TTSProcessor

from ..data.dataset_pretrain import Qwen3TTSPretrainDataset
from ..losses.dual_track_loss import Qwen3TTSDualTrackLoss


class Qwen3TTSPretrainTrainer(Trainer):
    """
    Trainer for Qwen3-TTS pre-training (S1/S2/S3 stages).
    
    This trainer:
    - Uses HuggingFace Trainer with FSDP support
    - Computes dual-track loss (Talker + MTP)
    - Supports gradient checkpointing for memory efficiency
    """
    
    def __init__(
        self,
        model: Qwen3TTSForConditionalGeneration,
        args: TrainingArguments,
        train_dataset: Qwen3TTSPretrainDataset,
        eval_dataset: Optional[Qwen3TTSPretrainDataset] = None,
        processor: Optional[Qwen3TTSProcessor] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs,
        )
        self.processor = processor
        self.loss_fn = Qwen3TTSDualTrackLoss(
            talker_weight=1.0,
            mtp_weight=0.3,
        )
    
    def compute_loss(
        self,
        model: Qwen3TTSForConditionalGeneration,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute dual-track loss (Talker + MTP).
        
        Args:
            model: Qwen3TTSForConditionalGeneration model
            inputs: Batch dict from dataloader
            return_outputs: Whether to return outputs
            num_items_in_batch: Number of items in batch (for logging)
        
        Returns:
            loss: Total loss
        """
        loss, metrics = self.loss_fn(model, inputs)
        
        # Log metrics
        if self.state.is_world_process_zero:
            for k, v in metrics.items():
                self.log({k: v})
        
        return (loss, None) if return_outputs else loss


def build_pretrain_trainer(
    stage: str,
    config_path: str,
    data_dir: str,
    output_dir: Optional[str] = None,
) -> Qwen3TTSPretrainTrainer:
    """
    Build pre-training trainer for a specific stage.
    
    Args:
        stage: Training stage (s1, s2, or s3)
        config_path: Path to model config or checkpoint
        data_dir: Path to training data directory
        output_dir: Output directory for checkpoints
    
    Returns:
        Qwen3TTSPretrainTrainer: Configured trainer
    """
    # Stage-specific hyperparameters
    stage_configs = {
        "s1": {
            "learning_rate": 1e-4,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 128,
            "warmup_ratio": 0.01,
        },
        "s2": {
            "learning_rate": 3e-5,
            "num_train_epochs": 2,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 64,
            "warmup_ratio": 0.005,
        },
        "s3": {
            "learning_rate": 1e-5,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 64,
            "warmup_ratio": 0.0,
        },
    }
    
    if stage not in stage_configs:
        raise ValueError(f"Unknown stage: {stage}. Must be one of {list(stage_configs.keys())}")
    
    stage_config = stage_configs[stage]
    output_dir = output_dir or f"outputs/checkpoints/pretrain/{stage}"
    
    # Load model
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        config_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    
    # Load processor
    processor = Qwen3TTSProcessor.from_pretrained(config_path)
    
    # Load datasets
    train_dataset = Qwen3TTSPretrainDataset(
        data_dir=f"{data_dir}/train",
        processor=processor,
        config=model.config,
    )
    eval_dataset = Qwen3TTSPretrainDataset(
        data_dir=f"{data_dir}/val",
        processor=processor,
        config=model.config,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=stage_config["num_train_epochs"],
        per_device_train_batch_size=stage_config["per_device_train_batch_size"],
        gradient_accumulation_steps=stage_config["gradient_accumulation_steps"],
        learning_rate=stage_config["learning_rate"],
        weight_decay=0.01,
        warmup_ratio=stage_config["warmup_ratio"],
        lr_scheduler_type="cosine",
        bf16=True,
        tf32=True,
        dataloader_num_workers=8,
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=3,
        logging_steps=10,
        gradient_checkpointing=True,
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "min_num_params": 1e7,
            "transformer_layer_cls_to_wrap": "Qwen3TTSTalkerDecoderLayer",
        },
        report_to="wandb",
        run_name=f"qwen3_tts_pretrain_{stage}",
        remove_unused_columns=False,
    )
    
    # Build trainer
    trainer = Qwen3TTSPretrainTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processor=processor,
    )
    
    return trainer


if __name__ == "__main__":
    import sys
    
    stage = sys.argv[1] if len(sys.argv) > 1 else "s1"
    config_path = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    data_dir = sys.argv[3] if len(sys.argv) > 3 else f"data/pretrain/{stage}"
    
    trainer = build_pretrain_trainer(stage, config_path, data_dir)
    trainer.train()