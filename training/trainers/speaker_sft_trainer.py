# training/trainers/speaker_sft_trainer.py
# -*- coding: utf-8 -*-
"""
Speaker SFT Trainer for Qwen3-TTS.

This trainer implements lightweight speaker fine-tuning for adopting
specific voices while improving naturalness and controllability.

This trainer wraps the original sft_12hz.py training logic.
"""
import json
import os
import shutil
from typing import Any, Dict, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from accelerate import Accelerator
from safetensors.torch import save_file

from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from qwen_tts.core.models.processing_qwen3_tts import Qwen3TTSProcessor

from ..data.dataset_pretrain import Qwen3TTSPretrainDataset
from ..losses.dual_track_loss import Qwen3TTSDualTrackLoss


class SpeakerSFTTrainer(Trainer):
    """
    Speaker SFT Trainer for Qwen3-TTS.
    
    This trainer:
    - Freezes backbone and speaker encoder
    - Trains text_projection, codec_head, and code_predictor
    - Injects target speaker embedding into codec_embedding
    """
    
    def __init__(
        self,
        model: Qwen3TTSForConditionalGeneration,
        args: TrainingArguments,
        train_dataset: Qwen3TTSPretrainDataset,
        eval_dataset: Optional[Qwen3TTSPretrainDataset] = None,
        processor: Optional[Qwen3TTSProcessor] = None,
        target_speaker: str = "aiden",
        target_speaker_id: int = 3000,
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
        self.target_speaker = target_speaker
        self.target_speaker_id = target_speaker_id
        self.target_speaker_embedding = None
        self.loss_fn = Qwen3TTSDualTrackLoss(
            talker_weight=1.0,
            mtp_weight=0.3,
        )
        
        # Freeze backbone and speaker encoder
        self._freeze_modules()
    
    def _freeze_modules(self):
        """Freeze backbone and speaker encoder, train adapter layers."""
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze adapter layers
        trainable_modules = [
            "text_projection",
            "codec_head",
            "code_predictor",
            "codec_embedding",
        ]
        
        for name, param in self.model.named_parameters():
            if any(module in name for module in trainable_modules):
                param.requires_grad = True
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"[SpeakerSFT] Target speaker: {self.target_speaker} (ID: {self.target_speaker_id})")
        print(f"[SpeakerSFT] Trainable parameters: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.2f}%)")
    
    def compute_loss(
        self,
        model: Qwen3TTSForConditionalGeneration,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute dual-track loss for speaker SFT.
        
        Args:
            model: Qwen3TTSForConditionalGeneration model
            inputs: Batch dict from dataloader
            return_outputs: Whether to return outputs
            num_items_in_batch: Number of items in batch
        
        Returns:
            loss: Total loss
        """
        loss, metrics = self.loss_fn(model, inputs)
        
        # Log metrics
        if self.state.is_world_process_zero:
            for k, v in metrics.items():
                self.log({k: v})
        
        return (loss, None) if return_outputs else loss
    
    def save_model(
        self,
        output_dir: Optional[str] = None,
        _internal_call: bool = False,
    ):
        """
        Save model with speaker embedding injection.
        
        Args:
            output_dir: Output directory
            _internal_call: Internal call flag
        """
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy original model files
        if os.path.isdir(self.args.output_dir):
            for item in os.listdir(self.args.output_dir):
                src = os.path.join(self.args.output_dir, item)
                dst = os.path.join(output_dir, item)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
        
        # Update config.json
        config_path = os.path.join(output_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            
            # Update tts_model_type
            config_dict["tts_model_type"] = "custom_voice"
            
            # Update talker_config
            talker_config = config_dict.get("talker_config", {})
            talker_config["spk_id"] = {self.target_speaker: self.target_speaker_id}
            talker_config["spk_is_dialect"] = {self.target_speaker: False}
            config_dict["talker_config"] = talker_config
            
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        # Save model weights with speaker embedding injection
        state_dict = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        
        # Drop speaker_encoder weights
        keys_to_drop = [k for k in state_dict.keys() if k.startswith("speaker_encoder")]
        for k in keys_to_drop:
            del state_dict[k]
        
        # Inject target speaker embedding
        if self.target_speaker_embedding is not None:
            weight_key = "talker.model.codec_embedding.weight"
            if weight_key in state_dict:
                state_dict[weight_key][self.target_speaker_id] = (
                    self.target_speaker_embedding.to(state_dict[weight_key].device)
                    .to(state_dict[weight_key].dtype)
                )
        
        # Save with safetensors
        save_path = os.path.join(output_dir, "model.safetensors")
        save_file(state_dict, save_path)
        
        print(f"[SpeakerSFT] Model saved to {output_dir}")
        print(f"[SpeakerSFT] Target speaker embedding injected at ID {self.target_speaker_id}")


def build_speaker_sft_trainer(
    config_path: str,
    data_dir: str,
    output_dir: Optional[str] = None,
    target_speaker: str = "aiden",
    target_speaker_id: int = 3000,
) -> SpeakerSFTTrainer:
    """
    Build speaker SFT trainer.
    
    Args:
        config_path: Path to model config or checkpoint
        data_dir: Path to training data directory
        output_dir: Output directory for checkpoints
        target_speaker: Target speaker name
        target_speaker_id: Target speaker ID
    
    Returns:
        SpeakerSFTTrainer: Configured speaker SFT trainer
    """
    output_dir = output_dir or f"outputs/checkpoints/speaker_sft/{target_speaker}"
    
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
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        tf32=True,
        dataloader_num_workers=4,
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=5,
        eval_strategy="epoch",
        gradient_checkpointing=True,
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "min_num_params": 1e7,
            "transformer_layer_cls_to_wrap": "Qwen3TTSTalkerDecoderLayer",
        },
        report_to="wandb",
        run_name=f"qwen3_tts_speaker_sft_{target_speaker}",
        remove_unused_columns=False,
    )
    
    # Build trainer
    trainer = SpeakerSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processor=processor,
        target_speaker=target_speaker,
        target_speaker_id=target_speaker_id,
    )
    
    return trainer


if __name__ == "__main__":
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/checkpoints/gspo/checkpoint-final"
    data_dir = sys.argv[2] if len(sys.argv) > 2 else "data/posttrain/speaker_sft"
    target_speaker = sys.argv[3] if len(sys.argv) > 3 else "aiden"
    
    trainer = build_speaker_sft_trainer(config_path, data_dir, target_speaker=target_speaker)
    trainer.train()