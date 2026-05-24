# training/losses/dual_track_loss.py
# -*- coding: utf-8 -*-
"""
Dual-Track Loss for Qwen3-TTS Pre-training (S1/S2/S3).

This loss function computes:
1. Talker loss: Cross-entropy loss for codebook 0 (main talker head)
2. MTP loss: Cross-entropy loss for codebooks 1-15 (Multi-Token Prediction)

Total loss = talker_weight * talker_loss + mtp_weight * mtp_loss

This implementation follows the architecture in sft_12hz.py:
- Talker predicts codebook 0 via codec_head
- MTP module predicts codebooks 1-15 via forward_sub_talker_finetune
"""
# training/losses/*.py
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class Qwen3TTSDualTrackLoss(nn.Module):
    """
    Dual-Track Loss for Qwen3-TTS pre-training.
    
    Computes:
    - Talker loss: CE loss for codebook 0 prediction
    - MTP loss: CE loss for codebooks 1-15 prediction
    
    Args:
        talker_weight: Weight for talker loss (default: 1.0)
        mtp_weight: Weight for MTP loss (default: 0.3, following sft_12hz.py)
    """
    
    def __init__(
        self,
        talker_weight: float = 1.0,
        mtp_weight: float = 0.3,
    ):
        super().__init__()
        self.talker_weight = talker_weight
        self.mtp_weight = mtp_weight
    
    def __call__(
        self,
        model,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute dual-track loss.
        
        Args:
            model: Qwen3TTSForConditionalGeneration model
            batch: Batch dict containing:
                - input_ids: (B, T, 2) - dual-channel input
                - codec_ids: (B, T, 16) - all 16 codebooks
                - codec_0_labels: (B, T) - labels for codebook 0
                - codec_mask: (B, T) - mask for valid codec positions
                - attention_mask: (B, T)
                - text_embedding_mask: (B, T, 1)
                - codec_embedding_mask: (B, T, 1)
                - ref_mels: (B, T_mel, mel_dim) - reference mel spectrograms
        
        Returns:
            total_loss: Total loss
            metrics: Dict with loss breakdown
        """
        # Unpack batch
        input_ids = batch['input_ids']
        codec_ids = batch['codec_ids']
        codec_0_labels = batch['codec_0_labels']
        codec_mask = batch['codec_mask']
        attention_mask = batch['attention_mask']
        text_embedding_mask = batch['text_embedding_mask']
        codec_embedding_mask = batch['codec_embedding_mask']
        ref_mels = batch['ref_mels']
        
        # Extract speaker embedding from reference mel
        speaker_embedding = model.speaker_encoder(
            ref_mels.to(model.device).to(model.dtype)
        ).detach()
        
        # Split input_ids into text and codec channels
        input_text_ids = input_ids[:, :, 0]
        input_codec_ids = input_ids[:, :, 1]
        
        # Compute embeddings
        input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
        input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
        
        # Inject speaker embedding at position 6 (following sft_12hz.py)
        input_codec_embedding[:, 6, :] = speaker_embedding
        
        # Dual-track: text + codec embeddings
        input_embeddings = input_text_embedding + input_codec_embedding
        
        # Add codec embeddings for codebooks 1-15
        for i in range(1, 16):
            codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](
                codec_ids[:, :, i]
            )
            codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
            input_embeddings = input_embeddings + codec_i_embedding
        
        # Forward through talker model
        outputs = model.talker(
            inputs_embeds=input_embeddings[:, :-1, :],
            attention_mask=attention_mask[:, :-1],
            labels=codec_0_labels[:, 1:],
            output_hidden_states=True
        )
        
        # Talker loss (codebook 0)
        talker_loss = outputs.loss
        
        # Extract hidden states for MTP
        hidden_states = outputs.hidden_states[0][-1]
        talker_hidden_states = hidden_states[codec_mask[:, :-1]]
        talker_codec_ids = codec_ids[codec_mask]
        
        # MTP loss (codebooks 1-15)
        sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(
            talker_codec_ids, 
            talker_hidden_states
        )
        
        # Total loss
        total_loss = self.talker_weight * talker_loss + self.mtp_weight * sub_talker_loss
        
        metrics = {
            'total_loss': total_loss.item(),
            'talker_loss': talker_loss.item(),
            'mtp_loss': sub_talker_loss.item(),
        }
        
        return total_loss, metrics


class Qwen3TTSSimpleLoss(nn.Module):
    """
    Simple loss wrapper that directly uses model's forward pass.
    This follows the exact implementation in sft_12hz.py.
    """
    
    def __init__(self, mtp_weight: float = 0.3):
        super().__init__()
        self.mtp_weight = mtp_weight
    
    def __call__(self, model, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss following sft_12hz.py implementation.
        """
        # Unpack batch
        input_ids = batch['input_ids']
        codec_ids = batch['codec_ids']
        ref_mels = batch['ref_mels']
        text_embedding_mask = batch['text_embedding_mask']
        codec_embedding_mask = batch['codec_embedding_mask']
        attention_mask = batch['attention_mask']
        codec_0_labels = batch['codec_0_labels']
        codec_mask = batch['codec_mask']
        
        # Extract speaker embedding
        speaker_embedding = model.speaker_encoder(
            ref_mels.to(model.device).to(model.dtype)
        ).detach()
        
        # Split channels
        input_text_ids = input_ids[:, :, 0]
        input_codec_ids = input_ids[:, :, 1]
        
        # Compute embeddings
        input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
        input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
        input_codec_embedding[:, 6, :] = speaker_embedding
        
        input_embeddings = input_text_embedding + input_codec_embedding
        
        # Add codebooks 1-15
        for i in range(1, 16):
            codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](
                codec_ids[:, :, i]
            )
            codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
            input_embeddings = input_embeddings + codec_i_embedding
        
        # Forward
        outputs = model.talker(
            inputs_embeds=input_embeddings[:, :-1, :],
            attention_mask=attention_mask[:, :-1],
            labels=codec_0_labels[:, 1:],
            output_hidden_states=True
        )
        
        # Extract hidden states
        hidden_states = outputs.hidden_states[0][-1]
        talker_hidden_states = hidden_states[codec_mask[:, :-1]]
        talker_codec_ids = codec_ids[codec_mask]
        
        # MTP forward
        sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(
            talker_codec_ids, 
            talker_hidden_states
        )
        
        # Total loss (following sft_12hz.py: loss = outputs.loss + 0.3 * sub_talker_loss)
        loss = outputs.loss + self.mtp_weight * sub_talker_loss
        
        metrics = {
            'loss': loss.item(),
            'talker_loss': outputs.loss.item(),
            'sub_talker_loss': sub_talker_loss.item(),
        }
        
        return loss, metrics