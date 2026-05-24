# training/rewards/rule_based_reward.py
# -*- coding: utf-8 -*-
"""
Rule-based reward combiner for Qwen3-TTS GSPO training.

Paper §3.2: "we employ rule-based rewards and leverage GSPO to comprehensively 
enhance the model's capabilities and stability across tasks"

Reward = w_wer * (1-WER) + w_sim * SIM + w_mos * UTMOS_normalized

This class combines multiple scorers (ASR, Speaker, UTMOS) into a single
reward signal for GSPO training.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple
import numpy as np

from .asr_scorer import ASRScorer
from .speaker_scorer import SpeakerSimilarityScorer
from .utmos_scorer import UTMOSQualityScorer


class Qwen3TTSRuleBasedReward(nn.Module):
    """
    Rule-based reward combiner for GSPO training.
    
    Combines multiple reward signals:
    - WER reward: content consistency (1 - WER)
    - SIM reward: speaker similarity
    - UTMOS reward: speech quality
    
    Total reward = w_wer * WER_reward + w_sim * SIM_reward + w_mos * UTMOS_reward
    
    Args:
        asr_model_name: ASR model name (default: "Qwen/Qwen3-ASR")
        speaker_model_name: Speaker model name (default: "microsoft/wavlm-large")
        utmos_model_name: UTMOS model name (default: "sarulab-speech/UTMOS22")
        device: Device for inference
        w_wer: Weight for WER reward (default: 1.0)
        w_sim: Weight for SIM reward (default: 0.5)
        w_mos: Weight for UTMOS reward (default: 0.3)
        wer_threshold: WER threshold above which reward = 0 (default: 0.3)
        utmos_threshold: UTMOS threshold below which reward is penalized (default: 3.0)
    """
    
    def __init__(
        self,
        asr_model_name: str = "Qwen/Qwen3-ASR",
        speaker_model_name: str = "microsoft/wavlm-large",
        utmos_model_name: str = "sarulab-speech/UTMOS22",
        device: str = "cuda",
        w_wer: float = 1.0,
        w_sim: float = 0.5,
        w_mos: float = 0.3,
        wer_threshold: float = 0.3,
        utmos_threshold: float = 3.0,
    ):
        super().__init__()
        
        self.w_wer = w_wer
        self.w_sim = w_sim
        self.w_mos = w_mos
        self.wer_threshold = wer_threshold
        self.utmos_threshold = utmos_threshold
        
        # Initialize scorers
        self.asr_scorer = ASRScorer(
            model_name=asr_model_name,
            device=device,
            wer_threshold=wer_threshold,
        )
        
        self.speaker_scorer = SpeakerSimilarityScorer(
            model_name=speaker_model_name,
            device=device,
        )
        
        self.utmos_scorer = UTMOSQualityScorer(
            model_name=utmos_model_name,
            device=device,
            utmos_threshold=utmos_threshold,
        )
    
    @torch.inference_mode()
    def compute_rewards(
        self,
        generated_audio: List[np.ndarray],
        reference_texts: List[str],
        reference_audio: Optional[List[np.ndarray]] = None,
        sample_rate: int = 24000,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined rewards for generated audio.
        
        Args:
            generated_audio: List of generated audio waveforms
            reference_texts: List of reference texts (for WER)
            reference_audio: List of reference audio waveforms (for SIM, optional)
            sample_rate: Sample rate of audio
            
        Returns:
            rewards: Tensor of shape (B,) with combined reward values
            metrics: Dict with all metrics
        """
        # Compute WER rewards
        wer_rewards, wer_metrics = self.asr_scorer.score(
            generated_audio,
            reference_texts,
            sample_rate,
        )
        
        # Compute SIM rewards (if reference audio provided)
        if reference_audio is not None:
            sim_rewards, sim_metrics = self.speaker_scorer.score(
                generated_audio,
                reference_audio,
                sample_rate,
            )
        else:
            # No reference audio, SIM reward = 0
            sim_rewards = torch.zeros(len(generated_audio), dtype=torch.float32)
            sim_metrics = {
                "sim_mean": 0.0,
                "sim_std": 0.0,
                "sim_reward_mean": 0.0,
                "sim_reward_std": 0.0,
            }
        
        # Compute UTMOS rewards
        utmos_rewards, utmos_metrics = self.utmos_scorer.score(
            generated_audio,
            sample_rate,
        )
        
        # Combine rewards
        combined_rewards = (
            self.w_wer * wer_rewards +
            self.w_sim * sim_rewards +
            self.w_mos * utmos_rewards
        )
        
        # Aggregate metrics
        metrics = {
            **wer_metrics,
            **sim_metrics,
            **utmos_metrics,
            "combined_reward_mean": combined_rewards.mean().item(),
            "combined_reward_std": combined_rewards.std().item(),
        }
        
        return combined_rewards, metrics
    
    def forward(
        self,
        generated_audio: List[np.ndarray],
        reference_texts: List[str],
        reference_audio: Optional[List[np.ndarray]] = None,
        sample_rate: int = 24000,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass for reward computation.
        
        Args:
            generated_audio: List of generated audio waveforms
            reference_texts: List of reference texts
            reference_audio: List of reference audio waveforms (optional)
            sample_rate: Sample rate of audio
            
        Returns:
            rewards: Tensor of shape (B,) with combined reward values
            metrics: Dict with all metrics
        """
        return self.compute_rewards(
            generated_audio,
            reference_texts,
            reference_audio,
            sample_rate,
        )