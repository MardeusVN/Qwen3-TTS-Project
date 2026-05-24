# training/data/dataset_pretrain.py
# -*- coding: utf-8 -*-
"""
Pretraining Dataset for Qwen3-TTS (S1/S2/S3 stages).

This dataset is designed for pretraining stages where:
- No reference audio is needed (unlike SFT)
- Multi-language support with sampling weights
- Long speech upsampling for S3 stage
- Probabilistic thinking pattern (random think/nothink)

Based on finetuning/dataset.py but extended for pretraining requirements.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import random
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig


AudioLike = Union[
    str,                     # wav path, URL, base64
    np.ndarray,              # waveform (requires sr)
    Tuple[np.ndarray, int],  # (waveform, sr)
]
MaybeList = Union[Any, List[Any]]


class Qwen3TTSPretrainDataset(Dataset):
    """
    Pretraining Dataset for Qwen3-TTS.
    
    Unlike SFT dataset, pretraining dataset:
    - Does NOT require ref_audio (no voice cloning during pretraining)
    - Supports multi-language with sampling weights
    - Supports long speech upsampling for S3 stage
    - Supports probabilistic thinking pattern (random think/nothink)
    
    JSONL format:
    {
        "audio": "path/to/audio.wav",
        "text": "transcript text",
        "language": "chinese",  # or "english", "japanese", etc.
        "duration": 10.5  # optional, for long speech upsampling
    }
    
    Args:
        data_list: List of data dictionaries
        processor: Text processor/tokenizer
        config: Qwen3TTSConfig
        stage: Training stage (s1/s2/s3)
        language_weights: Sampling weights for languages (optional)
        thinking_prob: Probability of using thinking pattern (default 0.3)
        lag_num: Lag number (default -1)
    """
    
    def __init__(
        self,
        data_list: List[Dict],
        processor,
        config: Qwen3TTSConfig,
        stage: str = "s1",
        language_weights: Optional[Dict[str, float]] = None,
        thinking_prob: float = 0.3,
        lag_num: int = -1,
    ):
        self.data_list = data_list
        self.processor = processor
        self.lag_num = lag_num
        self.config = config
        self.stage = stage
        self.thinking_prob = thinking_prob
        
        # Language sampling weights
        if language_weights is None:
            # Default weights from paper §3.2
            self.language_weights = {
                "chinese": 0.35,
                "english": 0.30,
                "german": 0.05,
                "italian": 0.05,
                "portuguese": 0.05,
                "spanish": 0.05,
                "japanese": 0.05,
                "korean": 0.05,
                "french": 0.05,
                "russian": 0.05,
            }
        else:
            self.language_weights = language_weights
        
        # Normalize weights
        total_weight = sum(self.language_weights.values())
        self.language_weights = {k: v / total_weight for k, v in self.language_weights.items()}
    
    def __len__(self):
        return len(self.data_list)
    
    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:
        """Load audio file to numpy array."""
        audio, sr = librosa.load(x, sr=None, mono=True)
        
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        
        return audio.astype(np.float32), int(sr)
    
    def _normalize_audio_inputs(self, audios: Union[AudioLike, List[AudioLike]]) -> List[Tuple[np.ndarray, int]]:
        """Normalize audio inputs into a list of (waveform, sr)."""
        if isinstance(audios, list):
            items = audios
        else:
            items = [audios]
        
        out: List[Tuple[np.ndarray, int]] = []
        for a in items:
            if isinstance(a, str):
                out.append(self._load_audio_to_np(a))
            elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                out.append((a[0].astype(np.float32), int(a[1])))
            elif isinstance(a, np.ndarray):
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                raise TypeError(f"Unsupported audio input type: {type(a)}")
        return out
    
    def _build_assistant_text(self, text: str) -> str:
        """Build assistant text in ChatML format."""
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    
    def _ensure_list(self, x: MaybeList) -> List[Any]:
        """Ensure input is a list."""
        return x if isinstance(x, list) else [x]
    
    def _tokenize_texts(self, text) -> List[torch.Tensor]:
        """Tokenize text to tensor."""
        input = self.processor(text=text, return_tensors="pt", padding=True)
        input_id = input["input_ids"]
        input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
        return input_id
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        """
        Get a single training sample.
        
        Returns:
            Dictionary with keys:
                - text_ids: Text token IDs
                - audio_codes: Audio codes (T, 16) for 12Hz
                - language: Language code
                - use_thinking: Whether to use thinking pattern
        """
        item = self.data_list[idx]
        
        audio_path = item["audio"]
        text = item["text"]
        language = item.get('language', 'auto')
        
        # Build assistant text
        text = self._build_assistant_text(text)
        text_ids = self._tokenize_texts(text)
        
        # Load audio
        audio_path_list = self._ensure_list(audio_path)
        normalized = self._normalize_audio_inputs(audio_path_list)
        wav, sr = normalized[0]
        
        # Encode audio to codes
        # Note: This assumes tokenizer is available and loaded separately
        # In practice, you would load tokenizer once and pass it to dataset
        # For now, we assume audio_codes are pre-computed in JSONL
        if "audio_codes" in item:
            audio_codes = item["audio_codes"]
            audio_codes = torch.tensor(audio_codes, dtype=torch.long)
        else:
            raise ValueError(
                "audio_codes not found in item. "
                "Please run prepare_data.py first to encode audio."
            )
        
        # Randomly decide whether to use thinking pattern
        # This implements "probabilistically activated thinking pattern" from paper §3.3
        use_thinking = random.random() < self.thinking_prob
        
        return {
            "text_ids": text_ids,
            "audio_codes": audio_codes,
            "language": language,
            "use_thinking": use_thinking,
        }
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """
        Collate function for DataLoader.
        
        This implements the 2-channel format (text + codec) as in finetuning/dataset.py
        but extended for pretraining requirements.
        
        Returns:
            Dictionary with keys:
                - input_ids: (B, T, 2) - 2-channel input (text + codec)
                - attention_mask: (B, T)
                - text_embedding_mask: (B, T)
                - codec_embedding_mask: (B, T)
                - codec_0_labels: (B, T) - labels for codebook 0
                - codec_ids: (B, T, 16) - all 16 codebooks
                - codec_mask: (B, T)
                - use_thinking: List[bool] - whether to use thinking pattern
        """
        assert self.lag_num == -1
        
        # Calculate max length
        item_length = [b['text_ids'].shape[1] + b['audio_codes'].shape[0] for b in batch]
        max_length = max(item_length) + 8
        b, t = len(batch), max_length
        
        # Initialize tensors (same as finetuning/dataset.py)
        input_ids = torch.zeros((b, t, 2), dtype=torch.long)
        codec_ids = torch.zeros((b, t, 16), dtype=torch.long)
        text_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
        codec_mask = torch.zeros((b, t), dtype=torch.bool)
        attention_mask = torch.zeros((b, t), dtype=torch.long)
        codec_0_labels = torch.full((b, t), -100, dtype=torch.long)
        
        use_thinking_list = []
        
        for i, data in enumerate(batch):
            text_ids = data['text_ids']
            audio_codec_0 = data['audio_codes'][:, 0]
            audio_codecs = data['audio_codes']
            use_thinking = data['use_thinking']
            
            text_ids_len = text_ids.shape[1]
            codec_ids_len = audio_codec_0.shape[0]
            
            # ========== Text channel (channel 0) ==========
            # Same as finetuning/dataset.py
            input_ids[i, :3, 0] = text_ids[0, :3]
            input_ids[i, 3:7, 0] = self.config.tts_pad_token_id
            input_ids[i, 7, 0] = self.config.tts_bos_token_id
            input_ids[i, 8:8 + text_ids_len - 3, 0] = text_ids[0, 3:]
            input_ids[i, 8 + text_ids_len - 3, 0] = self.config.tts_eos_token_id
            input_ids[i, 8 + text_ids_len - 2:8 + text_ids_len + codec_ids_len, 0] = (
                self.config.tts_pad_token_id
            )
            text_embedding_mask[i, :8 + text_ids_len + codec_ids_len] = True
            
            # ========== Codec channel (channel 1) ==========
            # Decide whether to use thinking pattern
            if use_thinking and data['language'] != 'auto':
                # Thinking pattern: codec_think_id + codec_think_bos_id + language_id + codec_think_eos_id
                language_id = self.config.talker_config.codec_language_id.get(
                    data['language'].lower(),
                    self.config.talker_config.codec_language_id['english']
                )
                codec_prefill = [
                    self.config.talker_config.codec_think_id,
                    self.config.talker_config.codec_think_bos_id,
                    language_id,
                    self.config.talker_config.codec_think_eos_id,
                    self.config.talker_config.codec_pad_id,
                ]
            else:
                # No-thinking pattern
                codec_prefill = [
                    self.config.talker_config.codec_nothink_id,
                    self.config.talker_config.codec_think_bos_id,
                    self.config.talker_config.codec_think_eos_id,
                    0,  # for speaker embedding (not used in pretraining)
                    self.config.talker_config.codec_pad_id,
                ]
            
            input_ids[i, 3:8, 1] = torch.tensor(codec_prefill)
            input_ids[i, 8:8 + text_ids_len - 3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8 + text_ids_len - 3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8 + text_ids_len - 2, 1] = self.config.talker_config.codec_bos_id
            input_ids[i, 8 + text_ids_len - 1:8 + text_ids_len - 1 + codec_ids_len, 1] = audio_codec_0
            input_ids[i, 8 + text_ids_len - 1 + codec_ids_len, 1] = (
                self.config.talker_config.codec_eos_token_id
            )
            
            # Labels for codebook 0
            codec_0_labels[i, 8 + text_ids_len - 1:8 + text_ids_len - 1 + codec_ids_len] = audio_codec_0
            codec_0_labels[i, 8 + text_ids_len - 1 + codec_ids_len] = (
                self.config.talker_config.codec_eos_token_id
            )
            
            # All 16 codebooks
            codec_ids[i, 8 + text_ids_len - 1:8 + text_ids_len - 1 + codec_ids_len, :] = audio_codecs
            
            # Masks
            codec_embedding_mask[i, 3:8 + text_ids_len + codec_ids_len] = True
            codec_embedding_mask[i, 3] = False  # for speaker embedding (not used in pretraining)
            
            codec_mask[i, 8 + text_ids_len - 1:8 + text_ids_len - 1 + codec_ids_len] = True
            attention_mask[i, :8 + text_ids_len + codec_ids_len] = True
            
            use_thinking_list.append(use_thinking)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'text_embedding_mask': text_embedding_mask.unsqueeze(-1),
            'codec_embedding_mask': codec_embedding_mask.unsqueeze(-1),
            'codec_0_labels': codec_0_labels,
            'codec_ids': codec_ids,
            'codec_mask': codec_mask,
            'use_thinking': use_thinking_list,
        }


def load_pretrain_dataset_from_jsonl(
    jsonl_path: str,
    processor,
    config: Qwen3TTSConfig,
    stage: str = "s1",
    language_weights: Optional[Dict[str, float]] = None,
    thinking_prob: float = 0.3,
) -> Qwen3TTSPretrainDataset:
    """
    Load pretraining dataset from JSONL file.
    
    JSONL format:
    {"audio": "path/to/audio.wav", "text": "transcript", "language": "chinese"}
    
    Args:
        jsonl_path: Path to JSONL file
        processor: Text processor/tokenizer
        config: Qwen3TTSConfig
        stage: Training stage (s1/s2/s3)
        language_weights: Sampling weights for languages
        thinking_prob: Probability of using thinking pattern
    
    Returns:
        Qwen3TTSPretrainDataset instance
    """
    data_list = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data_list.append(item)
    
    return Qwen3TTSPretrainDataset(
        data_list=data_list,
        processor=processor,
        config=config,
        stage=stage,
        language_weights=language_weights,
        thinking_prob=thinking_prob,
    )