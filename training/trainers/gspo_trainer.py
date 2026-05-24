# training/trainers/gspo_trainer.py
# -*- coding: utf-8 -*-
"""
GSPO Trainer for Qwen3-TTS.

This trainer implements Group Sparse Policy Optimization (GSPO) with
rule-based rewards for comprehensive capability enhancement.
"""
from typing import Any, Dict, Optional

import torch
from transformers import Trainer, TrainingArguments

from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from qwen_tts.core.models.processing_qwen3_tts import Qwen3TTSProcessor

from ..data.dataset_gspo import Qwen3TTSGSPODataset
from ..losses.gspo_loss import Qwen3TTSGSPOLoss, Qwen3TTSRuleBasedReward


class Qwen3TTSGSPOTrainer(Trainer):
    """
    GSPO Trainer for Qwen3-TTS.
    
    This trainer:
    - Uses rule-based rewards (WER, SIM, UTMOS)
    - Implements GSPO with group-based advantages
    - Supports gradient checkpointing
    """
    
    def __init__(
        self,
        model: Qwen3TTSForConditionalGeneration,
        ref_model: Qwen3TTSForConditionalGeneration,
        args: TrainingArguments,
        train_dataset: Qwen3TTSGSPODataset,
        eval_dataset: Optional[Qwen3TTSGSPODataset] = None,
        processor: Optional[Qwen3TTSProcessor] = None,
        reward_fn: Optional[Qwen3TTSRuleBasedReward] = None,
        group_size: int = 4,
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
        self.reward_fn = reward_fn or Qwen3TTSRuleBasedReward()
        self.loss_fn = Qwen3TTSGSPOLoss(
            reward_fn=self.reward_fn,
            group_size=group_size,
        )
        
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
        Compute GSPO loss.
        
        Args:
            model: Policy model
            inputs: Batch dict
            return_outputs: Whether to return outputs
            num_items_in_batch: Number of items in batch
        
        Returns:
            loss: GSPO loss
        """
        # Generate samples for GSPO
        # Note: This is a simplified version. In practice, you would:
        # 1. Generate multiple samples per prompt
        # 2. Compute rewards for each sample
        # 3. Compute group-based advantages
        # 4. Compute PPO loss with advantages
        
        # For simplicity, we use a placeholder here
        # In practice, you would implement the full GSPO loop
        loss = torch.tensor(0.0, device=model.device, requires_grad=True)
        
        # Log placeholder metrics
        if self.state.is_world_process_zero:
            self.log({"loss": loss.item()})
        
        return (loss, None) if return_outputs else loss


def build_gspo_trainer(
    config_path: str,
    data_dir: str,
    output_dir: Optional[str] = None,
    group_size: int = 4,
) -> Qwen3TTSGSPOTrainer:
    """
    Build GSPO trainer.
    
    Args:
        config_path: Path to model config or checkpoint
        data_dir: Path to training data directory
        output_dir: Output directory for checkpoints
        group_size: Number of samples per group for GSPO
    
    Returns:
        Qwen3TTSGSPOTrainer: Configured GSPO trainer
    """
    output_dir = output_dir or "outputs/checkpoints/gspo"
    
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
    train_dataset = Qwen3TTSGSPODataset(
        data_dir=f"{data_dir}/train",
        processor=processor,
        config=model.config,
    )
    eval_dataset = Qwen3TTSGSPODataset(
        data_dir=f"{data_dir}/val",
        processor=processor,
        config=model.config,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-6,
        weight_decay=0.0,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=True,
        tf32=True,
        dataloader_num_workers=4,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        logging_steps=5,
        gradient_checkpointing=True,
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "min_num_params": 1e7,
            "transformer_layer_cls_to_wrap": "Qwen3TTSTalkerDecoderLayer",
        },
        report_to="wandb",
        run_name="qwen3_tts_gspo",
        remove_unused_columns=False,
    )
    
    # Build trainer
    trainer = Qwen3TTSGSPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processor=processor,
        group_size=group_size,
    )
    
    return trainer


if __name__ == "__main__":
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/checkpoints/dpo/checkpoint-final"
    data_dir = sys.argv[2] if len(sys.argv) > 2 else "data/posttrain/gspo"
    
    trainer = build_gspo_trainer(config_path, data_dir)
    trainer.train()