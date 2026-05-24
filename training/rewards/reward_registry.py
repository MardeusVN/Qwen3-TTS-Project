# training/rewards/reward_registry.py
# -*- coding: utf-8 -*-
"""
Registry pattern for scorers and reward functions.

Allows flexible registration and combination of different scorers.
"""

from typing import Dict, Type, Callable, Any, List
import torch.nn as nn


class ScorerRegistry:
    """
    Registry for scorers.
    
    Allows dynamic registration and retrieval of scorer classes.
    """
    
    _registry: Dict[str, Type[nn.Module]] = {}
    
    @classmethod
    def register(cls, name: str, scorer_class: Type[nn.Module]) -> None:
        """
        Register a scorer class.
        
        Args:
            name: Name of the scorer
            scorer_class: Scorer class to register
        """
        if name in cls._registry:
            raise ValueError(f"Scorer '{name}' already registered")
        
        if not issubclass(scorer_class, nn.Module):
            raise ValueError(f"Scorer must be a subclass of nn.Module")
        
        cls._registry[name] = scorer_class
    
    @classmethod
    def get(cls, name: str) -> Type[nn.Module]:
        """
        Get a registered scorer class.
        
        Args:
            name: Name of the scorer
            
        Returns:
            Scorer class
        """
        if name not in cls._registry:
            raise ValueError(f"Scorer '{name}' not registered")
        
        return cls._registry[name]
    
    @classmethod
    def list_scorers(cls) -> List[str]:
        """List all registered scorers."""
        return list(cls._registry.keys())
    
    @classmethod
    def create(
        cls,
        name: str,
        **kwargs,
    ) -> nn.Module:
        """
        Create a scorer instance.
        
        Args:
            name: Name of the scorer
            **kwargs: Arguments to pass to scorer constructor
            
        Returns:
            Scorer instance
        """
        scorer_class = cls.get(name)
        return scorer_class(**kwargs)


# Register default scorers
from .asr_scorer import ASRScorer
from .speaker_scorer import SpeakerSimilarityScorer
from .utmos_scorer import UTMOSQualityScorer
from .rule_based_reward import Qwen3TTSRuleBasedReward

ScorerRegistry.register("asr", ASRScorer)
ScorerRegistry.register("speaker", SpeakerSimilarityScorer)
ScorerRegistry.register("utmos", UTMOSQualityScorer)
ScorerRegistry.register("rule_based", Qwen3TTSRuleBasedReward)