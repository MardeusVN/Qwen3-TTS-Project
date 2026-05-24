# training/trainers/dpo_trainer.py
# -*- coding: utf-8 -*-
"""
DPO Trainer for Qwen3-TTS.

This trainer implements Direct Preference Optimization (DPO) for aligning
model outputs with human preferences.
"""
from typing import Any, Dict, Optional

import torch
from transformers import Trainer, TrainingArguments

from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from qwen_tts.core.models.processing_qwen3_tts import Qwen3TTSProcessor

from ..data.dataset_dpo import Qwen3TTSDPODataset
from ..losses.dpo_loss import Qwen3TTSDPOLoss


class Qwen3TTSDPOTrainer(Trainer):
    """
    DPO Trainer for Qwen3-TTS.
    
    This trainer:
    - Uses HuggingFace Trainer
    - Computes DPO loss with reference model
    - Supports gradient checkpointing
    """
    
    def __init__(
        self,
        model: Qwen3TTSForConditionalGeneration,
        ref_model: Qwen3TTSForConditionalGeneration,
        args: TrainingArguments,
        train_dataset: Qwen3TTSDPODataset,
        eval_dataset: Optional[Qwen3TTSDPODataset] = None,
        processor: Optional[Qwen3TTSProcessor] = None,
        beta: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs,
        )
        self.ref_model = ref_model
        self.processor = processor
        self.loss_fn = Qwen3TTSDPOLoss(beta=beta)
        
        # Freeze reference model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def compute_loss(
        self,
        model: Qwen3TTSForConditionalGeneration,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute DPO loss.
        
        Args:
            model: Policy model
            inputs: Batch dict with chosen and rejected samples
            return_outputs: Whether to return outputs
            num_items_in_batch: Number of items in batch
        
        Returns:
            loss: DPO loss
        """
        loss, metrics = self.loss_fn(model, self.ref_model, inputs)
        
        # Log metrics
        if self.state.is_world_process_zero:
            for k, v in metrics.items():
                self.log({k: v})
        
        return (loss, None) if return_outputs else loss


def build_dpo_trainer(
    config_path: str,
    data_dir: str,
    output_dir: Optional[str] = None,
    beta: float = 0.1,
) -> Qwen3TTSDPOTrainer:
    """
    Build DPO trainer.
    
    Args:
        config_path: Path to model config or checkpoint
        data_dir: Path to training data directory
        output_dir: Output directory for checkpoints
        beta: DPO beta parameter
    
    Returns:
        Qwen3TTSDPOTrainer: Configured DPO trainer
    """
    output_dir = output_dir or "outputs/checkpoints/dpo"
    
    # Load policy model
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        config_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    
    # Load reference model (frozen copy)
    ref_model = Qwen3TTSForConditionalGeneration.from_pretrained(
        config_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    
    # Load processor
    processor = Qwen3TTSProcessor.from_pretrained(config_path)
    
    # Load datasets
    train_dataset = Qwen3TTSDPODataset(
        data_dir=f"{data_dir}/train",
        processor=processor,
        config=model.config,
    )
    eval_dataset = Qwen3TTSDPODataset(
        data_dir=f"{data_dir}/val",
        processor=processor,
        config=model.config,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=32,
        learning_rate=5e-6,
        weight_decay=0.0,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=True,
        tf32=True,
        dataloader_num_workers=4,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        logging_steps=5,
        gradient_checkpointing=True,
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "min_num_params": 1e7,
            "transformer_layer_cls_to_wrap": "Qwen3TTSTalkerDecoderLayer",
        },
        report_to="wandb",
        run_name="qwen3_tts_dpo",
        remove_unused_columns=False,
    )
    
    # Build trainer
    trainer = Qwen3TTSDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processor=processor,
        beta=beta,
    )
    
    return trainer


if __name__ == "__main__":
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/checkpoints/pretrain/s3/checkpoint-final"
    data_dir = sys.argv[2] if len(sys.argv) > 2 else "data/posttrain/dpo"
    
    trainer = build_dpo_trainer(config_path, data_dir)
    trainer.train()