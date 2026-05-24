# training/data/dataset_gspo.py
# -*- coding: utf-8 -*-
"""
GSPO Dataset for Qwen3-TTS.

This dataset is designed for Group Sparse Policy Optimization (GSPO) training
where we generate multiple responses for the same prompt and use rule-based rewards.

JSONL format:
{
    "prompt": "text prompt",
    "language": "chinese",
    "ref_audio": "ref.wav",  # optional, for speaker similarity reward
    "ref_audio_codes": [...]  # optional, pre-computed codes
}
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig


AudioLike = Union[
    str,
    np.ndarray,
    Tuple[np.ndarray, int],
]
MaybeList = Union[Any, List[Any]]


class Qwen3TTSGSPODataset(Dataset):
    """
    GSPO Dataset for Qwen3-TTS.
    
    Each sample contains:
    - prompt: Text prompt
    - language: Language code
    - ref_audio: Reference audio for speaker similarity (optional)
    - ref_audio_codes: Pre-computed reference audio codes (optional)
    
    During training, the model will generate multiple responses for each prompt,
    and rewards will be computed using rule-based metrics (WER, SIM, UTMOS).
    
    Args:
        data_list: List of data dictionaries
        processor: Text processor/tokenizer
        config: Qwen3TTSConfig
        group_size: Number of generations per prompt (default 4)
    """
    
    def __init__(
        self,
        data_list: List[Dict],
        processor,
        config: Qwen3TTSConfig,
        group_size: int = 4,
    ):
        self.data_list = data_list
        self.processor = processor
        self.config = config
        self.group_size = group_size
    
    def __len__(self):
        return len(self.data_list)
    
    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:
        """Load audio file to numpy array."""
        audio, sr = librosa.load(x, sr=None, mono=True)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        return audio.astype(np.float32), int(sr)
    
    def _normalize_audio_inputs(self, audios: Union[AudioLike, List[AudioLike]]) -> List[Tuple[np.ndarray, int]]:
        """Normalize audio inputs."""
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
        Get a single GSPO sample.
        
        Returns:
            Dictionary with keys:
                - text_ids: Text token IDs
                - language: Language code
                - ref_audio_codes: Reference audio codes (optional)
                - group_size: Number of generations per prompt
        """
        item = self.data_list[idx]
        
        prompt = item["prompt"]
        language = item.get('language', 'auto')
        
        # Build assistant text
        text = self._build_assistant_text(prompt)
        text_ids = self._tokenize_texts(text)
        
        # Load reference audio codes if available
        ref_audio_codes = None
        if "ref_audio_codes" in item:
            ref_audio_codes = torch.tensor(item["ref_audio_codes"], dtype=torch.long)
        elif "ref_audio" in item:
            # Load and encode reference audio
            ref_audio_path = item["ref_audio"]
            ref_audio_list = self._ensure_list(ref_audio_path)
            normalized = self._normalize_audio_inputs(ref_audio_list)
            ref_wav, ref_sr = normalized[0]
            
            # Note: In practice, you would encode this once and cache
            # For now, we assume ref_audio_codes are pre-computed
            raise ValueError(
                "ref_audio_codes not found. Please pre-compute reference audio codes."
            )
        
        return {
            "text_ids": text_ids,
            "language": language,
            "ref_audio_codes": ref_audio_codes,
            "group_size": self.group_size,
        }
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """
        Collate function for GSPO DataLoader.
        
        Returns batch with prompts and reference audio codes.
        """
        b = len(batch)
        
        # Calculate max text length
        max_text_len = max(b['text_ids'].shape[1] for b in batch)
        max_length = max_text_len + 8
        
        # Initialize tensors
        input_ids = torch.zeros((b, max_length, 2), dtype=torch.long)
        text_embedding_mask = torch.zeros((b, max_length), dtype=torch.bool)
        attention_mask = torch.zeros((b, max_length), dtype=torch.long)
        
        # Reference audio codes (optional)
        ref_audio_codes_list = []
        has_ref_audio = []
        
        languages = []
        group_sizes = []
        
        for i, data in enumerate(batch):
            text_ids = data['text_ids']
            language = data['language']
            ref_audio_codes = data['ref_audio_codes']
            group_size = data['group_size']
            
            text_ids_len = text_ids.shape[1]
            
            # Text channel
            input_ids[i, :3, 0] = text_ids[0, :3]
            input_ids[i, 3:7, 0] = self.config.tts_pad_token_id
            input_ids[i, 7, 0] = self.config.tts_bos_token_id
            input_ids[i, 8:8 + text_ids_len - 3, 0] = text_ids[0, 3:]
            input_ids[i, 8 + text_ids_len - 3, 0] = self.config.tts_eos_token_id
            input_ids[i, 8 + text_ids_len - 2:, 0] = self.config.tts_pad_token_id
            text_embedding_mask[i, :8 + text_ids_len] = True
            attention_mask[i, :8 + text_ids_len] = True
            
            # Codec channel (no-thinking pattern)
            codec_prefill = [
                self.config.talker_config.codec_nothink_id,
                self.config.talker_config.codec_think_bos_id,
                self.config.talker_config.codec_think_eos_id,
                0,  # for speaker embedding
                self.config.talker_config.codec_pad_id,
            ]
            
            input_ids[i, 3:8, 1] = torch.tensor(codec_prefill)
            input_ids[i, 8:8 + text_ids_len - 3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8 + text_ids_len - 3, 1] = self.config.talker_config.codec_pad_id
            
            # Reference audio codes
            if ref_audio_codes is not None:
                ref_audio_codes_list.append(ref_audio_codes)
                has_ref_audio.append(True)
            else:
                ref_audio_codes_list.append(None)
                has_ref_audio.append(False)
            
            languages.append(language)
            group_sizes.append(group_size)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'text_embedding_mask': text_embedding_mask.unsqueeze(-1),
            'ref_audio_codes': ref_audio_codes_list,
            'has_ref_audio': has_ref_audio,
            'languages': languages,
            'group_sizes': group_sizes,
        }


def load_gspo_dataset_from_jsonl(
    jsonl_path: str,
    processor,
    config: Qwen3TTSConfig,
    group_size: int = 4,
) -> Qwen3TTSGSPODataset:
    """
    Load GSPO dataset from JSONL file.
    
    JSONL format:
    {
        "prompt": "text prompt",
        "language": "chinese",
        "ref_audio": "ref.wav",  # optional
        "ref_audio_codes": [...]  # optional, pre-computed
    }
    
    Args:
        jsonl_path: Path to JSONL file
        processor: Text processor/tokenizer
        config: Qwen3TTSConfig
        group_size: Number of generations per prompt
    
    Returns:
        Qwen3TTSGSPODataset instance
    """
    data_list = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data_list.append(item)
    
    return Qwen3TTSGSPODataset(
        data_list=data_list,
        processor=processor,
        config=config,
        group_size=group_size,
    )