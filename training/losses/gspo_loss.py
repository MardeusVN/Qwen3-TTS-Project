# training/losses/gspo_loss.py
# -*- coding: utf-8 -*-
"""
Group Sparse Policy Optimization (GSPO) Loss for Qwen3-TTS.

Paper §3.2: "employ rule-based rewards and leverage GSPO to comprehensively 
enhance the model's capabilities and stability across tasks"

GSPO is a variant of PPO that uses rule-based rewards:
- WER reward: 1 - WER (content consistency)
- SIM reward: Cosine similarity (speaker similarity)
- UTMOS reward: Normalized UTMOS score (speech quality)

Total reward = w_wer * WER_reward + w_sim * SIM_reward + w_mos * UTMOS_reward
"""
# training/losses/*.py
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class Qwen3TTSRuleBasedReward:
    """
    Rule-based reward function for GSPO.
    
    Computes rewards based on:
    - WER: Word Error Rate (content consistency)
    - SIM: Speaker similarity (cosine similarity)
    - UTMOS: Speech quality score
    
    Args:
        w_wer: Weight for WER reward (default: 1.0)
        w_sim: Weight for SIM reward (default: 0.5)
        w_mos: Weight for UTMOS reward (default: 0.3)
        wer_threshold: WER threshold above which reward is 0 (default: 0.3)
        utmos_threshold: UTMOS threshold below which reward is penalized (default: 3.0)
    """
    
    def __init__(
        self,
        w_wer: float = 1.0,
        w_sim: float = 0.5,
        w_mos: float = 0.3,
        wer_threshold: float = 0.3,
        utmos_threshold: float = 3.0,
    ):
        self.w_wer = w_wer
        self.w_sim = w_sim
        self.w_mos = w_mos
        self.wer_threshold = wer_threshold
        self.utmos_threshold = utmos_threshold
        
        # Lazy load models
        self._asr_model = None
        self._spk_model = None
        self._utmos_model = None
    
    def _load_models(self, device: str = "cuda"):
        """Lazy load ASR, speaker verification, and UTMOS models."""
        if self._asr_model is None:
            # Lazy load ASR model (e.g., Qwen3-ASR or Whisper)
            # Placeholder: implement actual ASR model loading
            self._asr_model = None  # Placeholder
        
        if self._spk_model is None:
            # Lazy load speaker verification model (e.g., WavLM)
            # Placeholder: implement actual speaker verification model loading
            self._spk_model = None  # Placeholder
        
        if self._utmos_model is None:
            # Lazy load UTMOS model
            # Placeholder: implement actual UTMOS model loading
            self._utmos_model = None  # Placeholder
    
    def compute_wer_reward(
        self,
        generated_audio: torch.Tensor,
        reference_text: str,
        sample_rate: int = 24000,
    ) -> float:
        """
        Compute WER reward: 1 - WER.
        
        Args:
            generated_audio: Generated audio waveform
            reference_text: Reference text
            sample_rate: Sample rate
        
        Returns:
            WER reward in [0, 1]
        """
        # Placeholder: implement actual WER computation
        # Placeholder: use ASR model to transcribe generated audio
        # Placeholder: compute WER between transcription and reference_text
        
        # Placeholder implementation
        wer = 0.1  # Placeholder
        wer = min(wer, 1.0)
        reward = 1.0 - wer if wer < self.wer_threshold else 0.0
        
        return reward
    
    def compute_sim_reward(
        self,
        generated_audio: torch.Tensor,
        reference_audio: torch.Tensor,
        sample_rate: int = 24000,
    ) -> float:
        """
        Compute speaker similarity reward.
        
        Args:
            generated_audio: Generated audio waveform
            reference_audio: Reference audio waveform
            sample_rate: Sample rate
        
        Returns:
            SIM reward in [0, 1]
        """
        # Placeholder: implement actual speaker similarity computation
        # Placeholder: use speaker verification model to compute cosine similarity
        
        # Placeholder implementation
        sim = 0.8  # Placeholder
        reward = max(0.0, min(1.0, sim))
        
        return reward
    
    def compute_utmos_reward(
        self,
        generated_audio: torch.Tensor,
        sample_rate: int = 24000,
    ) -> float:
        """
        Compute UTMOS reward.
        
        Args:
            generated_audio: Generated audio waveform
            sample_rate: Sample rate
        
        Returns:
            UTMOS reward in [0, 1]
        """
        # Placeholder: implement actual UTMOS computation
        # Placeholder: use UTMOS model to predict MOS score
        
        # Placeholder implementation
        utmos = 4.0  # Placeholder (UTMOS range: 1-5)
        utmos_norm = (utmos - 1.0) / 4.0  # Normalize to [0, 1]
        utmos_norm = max(0.0, min(1.0, utmos_norm))
        
        # Penalize if below threshold
        if utmos < self.utmos_threshold:
            reward = utmos_norm * 0.5
        else:
            reward = utmos_norm
        
        return reward
    
    def __call__(
        self,
        generated_audio: torch.Tensor,
        reference_text: str,
        reference_audio: Optional[torch.Tensor] = None,
        sample_rate: int = 24000,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total reward.
        
        Args:
            generated_audio: Generated audio waveform
            reference_text: Reference text
            reference_audio: Reference audio waveform (optional)
            sample_rate: Sample rate
        
        Returns:
            reward: Total reward
            metrics: Dict with reward breakdown
        """
        # Lazy load models
        self._load_models(device=generated_audio.device)
        
        # Compute individual rewards
        wer_reward = self.compute_wer_reward(generated_audio, reference_text, sample_rate)
        sim_reward = self.compute_sim_reward(
            generated_audio, 
            reference_audio if reference_audio is not None else generated_audio,
            sample_rate
        )
        utmos_reward = self.compute_utmos_reward(generated_audio, sample_rate)
        
        # Total reward
        total_reward = (
            self.w_wer * wer_reward +
            self.w_sim * sim_reward +
            self.w_mos * utmos_reward
        )
        
        metrics = {
            'total_reward': total_reward,
            'wer_reward': wer_reward,
            'sim_reward': sim_reward,
            'utmos_reward': utmos_reward,
        }
        
        return torch.tensor(total_reward, device=generated_audio.device), metrics


class Qwen3TTSGSPOLoss(nn.Module):
    """
    Group Sparse Policy Optimization Loss for Qwen3-TTS.
    
    GSPO is a variant of PPO that uses group-based advantage estimation
    with rule-based rewards.
    
    Args:
        reward_fn: Rule-based reward function
        clip_range: PPO clip range (default: 0.2)
        kl_coef: KL divergence coefficient (default: 0.01)
        value_loss_coef: Value loss coefficient (default: 0.5)
    """
    
    def __init__(
        self,
        reward_fn: Qwen3TTSRuleBasedReward,
        clip_range: float = 0.2,
        kl_coef: float = 0.01,
        value_loss_coef: float = 0.5,
    ):
        super().__init__()
        self.reward_fn = reward_fn
        self.clip_range = clip_range
        self.kl_coef = kl_coef
        self.value_loss_coef = value_loss_coef
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        group_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute group-based advantages.
        
        Args:
            rewards: (B,) rewards for each sample
            group_size: Number of samples per group
        
        Returns:
            advantages: (B,) advantages
            baselines: (B,) baselines (group means)
        """
        B = rewards.shape[0]
        assert B % group_size == 0, f"Batch size {B} must be divisible by group_size {group_size}"
        
        num_groups = B // group_size
        rewards_grouped = rewards.view(num_groups, group_size)
        
        # Group mean (baseline)
        baselines = rewards_grouped.mean(dim=1, keepdim=True).expand(-1, group_size)
        baselines = baselines.reshape(-1)
        
        # Group std for normalization
        group_std = rewards_grouped.std(dim=1, keepdim=True).expand(-1, group_size)
        group_std = group_std.reshape(-1).clamp(min=1e-8)
        
        # Advantages = (reward - baseline) / std
        advantages = (rewards - baselines) / group_std
        
        return advantages, baselines
    
    def compute_policy_loss(
        self,
        logp_new: torch.Tensor,
        logp_old: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute PPO-clip policy loss.
        
        Args:
            logp_new: (B,) log probabilities under new policy
            logp_old: (B,) log probabilities under old policy
            advantages: (B,) advantages
        
        Returns:
            policy_loss: Policy loss
            metrics: Dict with loss breakdown
        """
        # Log ratio: log(π_new/π_old)
        log_ratio = logp_new - logp_old
        ratio = torch.exp(log_ratio)
        
        # PPO-clip
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # KL divergence (approximation: ratio - log(ratio) - 1)
        kl_div = (ratio - log_ratio - 1.0).mean()
        
        # Clip fraction
        clip_frac = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
        
        metrics = {
            'policy_loss': policy_loss.item(),
            'kl_div': kl_div.item(),
            'clip_frac': clip_frac,
            'ratio_mean': ratio.mean().item(),
        }
        
        return policy_loss, metrics
    
    def __call__(
        self,
        model,
        ref_model,
        batch: Dict[str, torch.Tensor],
        rewards: torch.Tensor,
        logp_old: torch.Tensor,
        group_size: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GSPO loss.
        
        Args:
            model: Policy model (being trained)
            ref_model: Reference model (frozen)
            batch: Batch dict
            rewards: (B,) rewards for each sample
            logp_old: (B,) log probabilities under old policy
            group_size: Number of samples per group
        
        Returns:
            loss: Total GSPO loss
            metrics: Dict with loss breakdown
        """
        # Compute group-based advantages
        advantages, baselines = self.compute_advantages(rewards, group_size)
        
        # Compute log probabilities under new policy
        # Placeholder: implement actual log probability computation
        # Placeholder: use model to compute log probabilities
        logp_new = logp_old  # Placeholder
        
        # Policy loss
        policy_loss, policy_metrics = self.compute_policy_loss(
            logp_new, logp_old, advantages
        )
        
        # KL divergence with reference model
        with torch.no_grad():
            # Placeholder: compute log probabilities under reference model
            logp_ref = logp_old  # Placeholder
        kl_div = (logp_new - logp_ref).mean()
        
        # Value loss (placeholder)
        value_loss = torch.tensor(0.0, device=logp_new.device)
        
        # Total loss
        total_loss = (
            policy_loss +
            self.kl_coef * kl_div +
            self.value_loss_coef * value_loss
        )
        
        metrics = {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'kl_div': kl_div.item(),
            'value_loss': value_loss.item(),
            'reward_mean': rewards.mean().item(),
            'baseline_mean': baselines.mean().item(),
            'advantage_mean': advantages.mean().item(),
            'advantage_std': advantages.std().item(),
            **policy_metrics,
        }
        
        return total_loss, metrics