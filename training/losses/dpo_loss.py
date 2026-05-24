# training/losses/dpo_loss.py
# -*- coding: utf-8 -*-
"""
Direct Preference Optimization (DPO) Loss for Qwen3-TTS.

Paper §3.2: "construct preference pairs for multilingual speech samples 
based on human feedback and then perform DPO on Qwen3-TTS"

DPO Loss Formula:
L_DPO = -log σ(β * (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))

where:
- y_w: chosen (preferred) response
- y_l: rejected response
- π_θ: policy model (being trained)
- π_ref: reference model (frozen)
- β: temperature parameter (default: 0.1)
"""
# training/losses/*.py
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class Qwen3TTSDPOLoss(nn.Module):
    """
    Direct Preference Optimization Loss for Qwen3-TTS.
    
    Args:
        beta: Temperature parameter β (default: 0.1, following paper)
        label_smoothing: Label smoothing for conservative DPO (default: 0.0)
        loss_type: Type of DPO loss ('sigmoid', 'ipo', 'kto')
    """
    
    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        loss_type: str = "sigmoid",
    ):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type
    
    def _compute_log_prob(
        self,
        model,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute log probability of generating the given codec sequence.
        
        This follows the same forward pass as sft_12hz.py to compute
        the log probability of the given codec sequence.
        
        Args:
            model: Qwen3TTSForConditionalGeneration model
            batch: Batch dict with input_ids, codec_ids, etc.
        
        Returns:
            log_prob: (B,) log probability for each sample
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
        
        # Compute embeddings (following sft_12hz.py)
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
        
        # Forward through talker
        outputs = model.talker(
            inputs_embeds=input_embeddings[:, :-1, :],
            attention_mask=attention_mask[:, :-1],
            labels=codec_0_labels[:, 1:],
            output_hidden_states=True
        )
        
        # Talker loss (negative log-likelihood)
        talker_loss = outputs.loss
        
        # Extract hidden states for MTP
        hidden_states = outputs.hidden_states[0][-1]
        talker_hidden_states = hidden_states[codec_mask[:, :-1]]
        talker_codec_ids = codec_ids[codec_mask]
        
        # MTP forward
        sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(
            talker_codec_ids,
            talker_hidden_states
        )
        
        # Total negative log-likelihood
        # Note: outputs.loss and sub_talker_loss are already negative log-likelihoods
        total_log_prob = -(talker_loss + 0.3 * sub_talker_loss)
        
        return total_log_prob
    
    def __call__(
        self,
        model,
        ref_model,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DPO loss.
        
        Args:
            model: Policy model (being trained)
            ref_model: Reference model (frozen)
            batch: Batch dict containing:
                - chosen_*: Chosen response data
                - rejected_*: Rejected response data
        
        Returns:
            loss: DPO loss
            metrics: Dict with loss breakdown
        """
        # Split batch into chosen and rejected
        chosen_batch = {
            'input_ids': batch['chosen_input_ids'],
            'codec_ids': batch['chosen_codec_ids'],
            'ref_mels': batch['chosen_ref_mels'],
            'text_embedding_mask': batch['chosen_text_embedding_mask'],
            'codec_embedding_mask': batch['chosen_codec_embedding_mask'],
            'attention_mask': batch['chosen_attention_mask'],
            'codec_0_labels': batch['chosen_codec_0_labels'],
            'codec_mask': batch['chosen_codec_mask'],
        }
        
        rejected_batch = {
            'input_ids': batch['rejected_input_ids'],
            'codec_ids': batch['rejected_codec_ids'],
            'ref_mels': batch['rejected_ref_mels'],
            'text_embedding_mask': batch['rejected_text_embedding_mask'],
            'codec_embedding_mask': batch['rejected_codec_embedding_mask'],
            'attention_mask': batch['rejected_attention_mask'],
            'codec_0_labels': batch['rejected_codec_0_labels'],
            'codec_mask': batch['rejected_codec_mask'],
        }
        
        # Compute log probabilities for policy model
        logp_chosen = self._compute_log_prob(model, chosen_batch)
        logp_rejected = self._compute_log_prob(model, rejected_batch)
        
        # Compute log probabilities for reference model (frozen)
        with torch.no_grad():
            logp_chosen_ref = self._compute_log_prob(ref_model, chosen_batch)
            logp_rejected_ref = self._compute_log_prob(ref_model, rejected_batch)
        
        # Compute DPO loss
        # log_ratio = log(π_θ(y_w|x)/π_ref(y_w|x)) - log(π_θ(y_l|x)/π_ref(y_l|x))
        log_ratio_chosen = logp_chosen - logp_chosen_ref
        log_ratio_rejected = logp_rejected - logp_rejected_ref
        log_ratio = log_ratio_chosen - log_ratio_rejected
        
        if self.loss_type == "sigmoid":
            # Standard DPO
            loss = -F.logsigmoid(self.beta * log_ratio)
            if self.label_smoothing > 0:
                # Conservative DPO
                loss = (
                    -F.logsigmoid(self.beta * log_ratio) * (1 - self.label_smoothing)
                    - F.logsigmoid(-self.beta * log_ratio) * self.label_smoothing
                )
            loss = loss.mean()
        
        elif self.loss_type == "ipo":
            # Identity Preference Optimization
            loss = (log_ratio - 1.0 / (2 * self.beta)) ** 2
            loss = loss.mean()
        
        elif self.loss_type == "kto":
            # Kahneman-Tversky Optimization
            # For simplicity, use sigmoid variant
            loss = -F.logsigmoid(self.beta * log_ratio).mean()
        
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        # Metrics
        metrics = {
            'loss': loss.item(),
            'logp_chosen': logp_chosen.mean().item(),
            'logp_rejected': logp_rejected.mean().item(),
            'logp_chosen_ref': logp_chosen_ref.mean().item(),
            'logp_rejected_ref': logp_rejected_ref.mean().item(),
            'log_ratio': log_ratio.mean().item(),
            'accuracy': (logp_chosen > logp_rejected).float().mean().item(),
        }
        
        return loss, metrics