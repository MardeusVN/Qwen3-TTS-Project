# training/rewards/utmos_scorer.py
# -*- coding: utf-8 -*-
"""
UTMOS-based scorer for speech quality.

Computes UTMOS (UTokyo-SaruLab MOS Predictor) score for speech quality.
Higher UTMOS = higher reward.

UTMOS score range: 1.0 (worst) to 5.0 (best)
Normalized reward: (UTMOS - 1.0) / 4.0
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple
import numpy as np


class UTMOSQualityScorer(nn.Module):
    """
    UTMOS-based scorer for speech quality.
    
    Computes UTMOS score for generated speech quality.
    Reward = (UTMOS - 1.0) / 4.0 (normalized to [0, 1])
    
    Args:
        model_name: UTMOS model name (default: "sarulab-speech/UTMOS22")
        device: Device for inference
        utmos_threshold: UTMOS threshold below which reward is penalized (default: 3.0)
    """
    
    def __init__(
        self,
        model_name: str = "sarulab-speech/UTMOS22",
        device: str = "cuda",
        utmos_threshold: float = 3.0,
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.utmos_threshold = utmos_threshold
        self.utmos_model = None
        self._load_model()
    
    def _load_model(self):
        """Load UTMOS model lazily."""
        if self.utmos_model is None:
            try:
                # Try to load UTMOS from HuggingFace
                try:
                    from utmos import UTMOS
                    self.utmos_model = UTMOS.from_pretrained(self.model_name).to(self.device)
                except ImportError:
                    print(f"[UTMOSScorer] utmos package not installed. Using fallback UTMOS.")
                    # Fallback: use a simple quality estimator
                    print("[UTMOSScorer] Using simple quality estimator as fallback.")
                    self.utmos_model = None
            except Exception as e:
                print(f"[UTMOSScorer] Failed to load UTMOS: {e}")
                self.utmos_model = None
    
    def _compute_utmos(
        self,
        audio: np.ndarray,
        sample_rate: int = 24000,
    ) -> float:
        """
        Compute UTMOS score for audio.
        
        Args:
            audio: Audio waveform (numpy array)
            sample_rate: Sample rate
            
        Returns:
            UTMOS score (1.0 to 5.0)
        """
        if self.utmos_model is None:
            # Fallback: simple quality estimator based on SNR and clarity
            # This is a simplified fallback - in production, use proper UTMOS
            snr = self._estimate_snr(audio)
            # Simple mapping: SNR > 20dB -> 4.0, SNR < 10dB -> 2.0
            snr_normalized = max(0.0, min(1.0, (snr - 10.0) / 10.0))
            utmos = 2.0 + snr_normalized * 2.0
            return float(utmos)
        
        # Use UTMOS model
        with torch.no_grad():
            audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)
            utmos_score = self.utmos_model(audio_tensor, sample_rate).item()
            return float(utmos_score)
    
    def _estimate_snr(self, audio: np.ndarray) -> float:
        """
        Estimate Signal-to-Noise Ratio (SNR) of audio.
        
        Simple SNR estimation based on signal energy vs noise energy.
        """
        # Estimate noise floor from quietest 10% of frames
        frame_length = int(0.025 * 24000)  # 25ms frames
        hop_length = int(0.010 * 24000)    # 10ms hop
        
        # Compute frame energies
        energies = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energy = np.sqrt(np.mean(frame ** 2))
            energies.append(energy)
        
        if not energies:
            return 20.0  # Default SNR
        
        energies = np.array(energies)
        
        # Noise floor: 10th percentile
        noise_floor = np.percentile(energies, 10)
        
        # Signal: 90th percentile
        signal_level = np.percentile(energies, 90)
        
        # SNR in dB
        snr = 20 * np.log10(signal_level / (noise_floor + 1e-8) + 1e-8)
        
        return float(snr)
    
    @torch.inference_mode()
    def score(
        self,
        generated_audio: List[np.ndarray],
        sample_rate: int = 24000,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute UTMOS-based rewards.
        
        Args:
            generated_audio: List of generated audio waveforms
            sample_rate: Sample rate of audio
            
        Returns:
            rewards: Tensor of shape (B,) with reward values
            metrics: Dict with metrics
        """
        rewards = []
        utmos_scores = []
        
        for audio in generated_audio:
            # Compute UTMOS score
            utmos = self._compute_utmos(audio, sample_rate)
            utmos_scores.append(utmos)
            
            # Normalize UTMOS to [0, 1]: (UTMOS - 1.0) / 4.0
            utmos_normalized = (utmos - 1.0) / 4.0
            utmos_normalized = max(0.0, min(1.0, utmos_normalized))
            
            # Apply threshold penalty
            if utmos < self.utmos_threshold:
                reward = utmos_normalized * 0.5  # Penalize low quality
            else:
                reward = utmos_normalized
            
            rewards.append(reward)
        
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        metrics = {
            "utmos_mean": np.mean(utmos_scores),
            "utmos_std": np.std(utmos_scores),
            "utmos_reward_mean": np.mean(rewards),
            "utmos_reward_std": np.std(rewards),
        }
        
        return rewards_tensor, metrics