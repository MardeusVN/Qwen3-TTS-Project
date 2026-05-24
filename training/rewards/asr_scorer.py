# training/rewards/asr_scorer.py
# -*- coding: utf-8 -*-
"""
ASR-based scorer for content consistency.

Computes Word Error Rate (WER) between generated speech transcription
and reference text. Lower WER = higher reward.

Reward = 1 - WER (capped at 0)

Supports:
- Qwen3-ASR (recommended, paper §4.2.6)
- Whisper (fallback)
- Wav2Vec2 (fallback)
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict
import numpy as np


class ASRScorer(nn.Module):
    """
    ASR-based scorer for content consistency.
    
    Computes WER between generated speech and reference text.
    Reward = max(0, 1 - WER)
    
    Args:
        model_name: ASR model name (default: "Qwen/Qwen3-ASR")
        device: Device for inference
        wer_threshold: WER threshold above which reward = 0 (default: 0.3)
        device: Device for inference
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-ASR",
        device: str = "cuda",
        wer_threshold: float = 0.3,
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.wer_threshold = wer_threshold
        self.asr_model = None
        self.asr_processor = None
        self._load_model()
    
    def _load_model(self):
        """Load ASR model lazily."""
        if self.asr_model is None:
            try:
                from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
                
                # Try to load Qwen3-ASR first
                try:
                    self.asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.bfloat16,
                        device_map=self.device,
                    )
                    self.asr_processor = AutoProcessor.from_pretrained(self.model_name)
                except Exception as e:
                    print(f"[ASRScorer] Failed to load {self.model_name}, falling back to Whisper: {e}")
                    # Fallback to Whisper
                    self.asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        "openai/whisper-large-v3",
                        torch_dtype=torch.bfloat16,
                        device_map=self.device,
                    )
                    self.asr_processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
            except ImportError:
                raise ImportError("transformers not installed. Install with: pip install transformers")
    
    def _compute_wer(self, hypothesis: str, reference: str) -> float:
        """
        Compute Word Error Rate between hypothesis and reference.
        
        Args:
            hypothesis: Transcribed text
            reference: Reference text
            
        Returns:
            WER score (0.0 = perfect, 1.0 = completely wrong)
        """
        try:
            import jiwer
            wer = jiwer.wer(reference.lower(), hypothesis.lower())
            return min(wer, 1.0)
        except ImportError:
            # Fallback: simple word-level edit distance
            hyp_words = hypothesis.lower().split()
            ref_words = reference.lower().split()
            
            if not ref_words:
                return 1.0 if hyp_words else 0.0
            
            # Simple word-level edit distance
            m, n = len(ref_words), len(hyp_words)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if ref_words[i - 1] == hyp_words[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1]
                    else:
                        dp[i][j] = 1 + min(
                            dp[i - 1][j],      # deletion
                            dp[i][j - 1],      # insertion
                            dp[i - 1][j - 1]   # substitution
                        )
            
            return min(dp[m][n] / m, 1.0)
    
    @torch.inference_mode()
    def transcribe(
        self,
        audio_waveforms: List[np.ndarray],
        sample_rate: int = 24000,
    ) -> List[str]:
        """
        Transcribe audio waveforms to text.
        
        Args:
            audio_waveforms: List of audio waveforms (numpy arrays)
            sample_rate: Sample rate of audio
            
        Returns:
            List of transcribed texts
        """
        if self.asr_model is None:
            self._load_model()
        
        transcriptions = []
        
        for audio in audio_waveforms:
            # Prepare input for ASR model
            inputs = self.asr_processor(
                audio,
                sampling_rate=sample_rate,
                return_tensors="pt",
                sampling_rate=sample_rate,
            ).to(self.device)
            
            # Transcribe
            with torch.no_grad():
                output_ids = self.asr_model.generate(
                    inputs.input_features,
                    max_new_tokens=512,
                )
            
            transcription = self.asr_processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
            )[0]
            transcriptions.append(transcription)
        
        return transcriptions
    
    @torch.inference_mode()
    def score(
        self,
        generated_audio: List[np.ndarray],
        reference_texts: List[str],
        sample_rate: int = 24000,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute WER-based rewards.
        
        Args:
            generated_audio: List of generated audio waveforms
            reference_texts: List of reference texts
            sample_rate: Sample rate of audio
            
        Returns:
            rewards: Tensor of shape (B,) with reward values
            metrics: Dict with metrics
        """
        # Transcribe generated audio
        transcriptions = self.transcribe(generated_audio, sample_rate)
        
        # Compute WER for each sample
        wer_scores = []
        for hyp, ref in zip(transcriptions, reference_texts):
            wer = self._compute_wer(hyp, ref)
            wer_scores.append(wer)
        
        # Convert WER to reward: reward = max(0, 1 - WER)
        rewards = []
        for wer in wer_scores:
            if wer > self.wer_threshold:
                reward = 0.0
            else:
                reward = max(0.0, 1.0 - wer)
            rewards.append(reward)
        
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        metrics = {
            "wer_mean": np.mean(wer_scores),
            "wer_std": np.std(wer_scores),
            "wer_reward_mean": np.mean(rewards),
            "wer_reward_std": np.std(rewards),
        }
        
        return rewards_tensor, metrics