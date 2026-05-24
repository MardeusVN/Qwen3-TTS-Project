# training/rewards/__init__.py
# -*- coding: utf-8 -*-
"""
Reward functions for Qwen3-TTS GSPO training.

This module provides rule-based reward functions as described in paper §3.2:
"we employ rule-based rewards and leverage GSPO to comprehensively enhance 
the model's capabilities and stability across tasks"

Reward = w_wer * (1-WER) + w_sim * SIM + w_mos * UTMOS_normalized
"""

from .asr_scorer import ASRScorer
from .speaker_scorer import SpeakerSimilarityScorer
from .utmos_scorer import UTMOSQualityScorer
from .rule_based_reward import Qwen3TTSRuleBasedReward
from .reward_registry import ScorerRegistry

__all__ = [
    "ASRScorer",
    "SpeakerSimilarityScorer",
    "UTMOSQualityScorer",
    "Qwen3TTSRuleBasedReward",
    "ScorerRegistry",
]