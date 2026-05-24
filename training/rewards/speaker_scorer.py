# training/rewards/speaker_scorer.py
# -*- coding: utf-8 -*-
"""
Speaker similarity scorer for voice consistency.

Computes cosine similarity between speaker embeddings of generated speech
and reference speech. Higher similarity = higher reward.

Supports:
- WavLM-large (recommended)
- ECAPA-TDNN (fallback)
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple
import numpy as np


class SpeakerSimilarityScorer(nn.Module):
    """
    Speaker similarity scorer.
    
    Computes cosine similarity between speaker embeddings of generated
    and reference speech.
    
    Args:
        model_name: Speaker verification model (default: "microsoft/wavlm-large")
        device: Device for inference
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/wavlm-large",
        device: str = "cuda",
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.speaker_model = None
        self.speaker_processor = None
        self._load_model()
    
    def _load_model(self):
        """Load speaker verification model lazily."""
        if self.speaker_model is None:
            try:
                from transformers import AutoModel, AutoFeatureExtractor
                
                # Try to load WavLM-large
                try:
                    self.speaker_model = AutoModel.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.bfloat16,
                        device_map=self.device,
                    )
                    self.speaker_processor = AutoFeatureExtractor.from_pretrained(self.model_name)
                except Exception as e:
                    print(f"[SpeakerScorer] Failed to load {self.model_name}, falling back to ECAPA-TDNN: {e}")
                    # Fallback to ECAPA-TDNN from Qwen3-TTS
                    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder
                    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig
                    
                    self.speaker_model = Qwen3TTSSpeakerEncoder(
                        Qwen3TTSSpeakerEncoderConfig()
                    ).to(self.device).to(torch.bfloat16)
                    self.speaker_processor = None
            except ImportError:
                raise ImportError("transformers not installed. Install with: pip install transformers")
    
    def _compute_speaker_embedding(
        self,
        audio: np.ndarray,
        sample_rate: int = 24000,
    ) -> np.ndarray:
        """
        Compute speaker embedding for audio.
        
        Args:
            audio: Audio waveform (numpy array)
            sample_rate: Sample rate
            
        Returns:
            Speaker embedding (numpy array)
        """
        with torch.no_grad():
            # Prepare input
            if self.speaker_processor is not None:
                inputs = self.speaker_processor(
                    audio,
                    sampling_rate=sample_rate,
                    return_tensors="pt",
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.speaker_model(**inputs)
                    # Mean pooling over time
                    embedding = outputs.last_hidden_state.mean(dim=1)
            else:
                # Fallback: use Qwen3TTSSpeakerEncoder
                from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
                
                audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)
                mel = mel_spectrogram(
                    audio_tensor,
                    n_fft=1024,
                    num_mels=128,
                    sampling_rate=24000,
                    hop_size=256,
                    win_size=1024,
                    fmin=0,
                    fmax=12000,
                ).transpose(1, 2)
                
                embedding = self.speaker_model(mel.to(self.speaker_model.fc.weight.dtype))
            
            return embedding.cpu().numpy()
    
    def _cosine_similarity(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray,
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        emb1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
        return float(np.dot(emb1, emb2))
    
    @torch.inference_mode()
    def score(
        self,
        generated_audio: List[np.ndarray],
        reference_audio: List[np.ndarray],
        sample_rate: int = 24000,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute speaker similarity rewards.
        
        Args:
            generated_audio: List of generated audio waveforms
            reference_audio: List of reference audio waveforms
            sample_rate: Sample rate of audio
            
        Returns:
            rewards: Tensor of shape (B,) with reward values
            metrics: Dict with metrics
        """
        rewards = []
        similarities = []
        
        for gen_audio, ref_audio in zip(generated_audio, reference_audio):
            # Compute speaker embeddings
            gen_emb = self._compute_speaker_embedding(gen_audio, sample_rate)
            ref_emb = self._compute_speaker_embedding(ref_audio, sample_rate)
            
            # Compute cosine similarity
            similarity = self._cosine_similarity(gen_emb, ref_emb)
            similarities.append(similarity)
            
            # Reward is similarity (already in [0, 1] range)
            reward = max(0.0, min(1.0, similarity))
            rewards.append(reward)
        
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        metrics = {
            "sim_mean": np.mean(similarities),
            "sim_std": np.std(similarities),
            "sim_reward_mean": np.mean(rewards),
            "sim_reward_std": np.std(rewards),
        }
        
        return rewards_tensor, metrics