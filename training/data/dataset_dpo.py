# training/data/dataset_dpo.py
# -*- coding: utf-8 -*-
"""
DPO Dataset for Qwen3-TTS.

This dataset is designed for Direct Preference Optimization (DPO) training
where we have chosen/rejected pairs for the same prompt.

JSONL format:
{
    "prompt": "text prompt",
    "chosen": {"audio": "chosen.wav", "audio_codes": [...]},
    "rejected": {"audio": "rejected.wav", "audio_codes": [...]},
    "language": "chinese"
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


class Qwen3TTSDPODataset(Dataset):
    """
    DPO Dataset for Qwen3-TTS.
    
    Each sample contains:
    - prompt: Text prompt (same for chosen and rejected)
    - chosen: Chosen response (audio + audio_codes)
    - rejected: Rejected response (audio + audio_codes)
    - language: Language code
    
    Args:
        data_list: List of data dictionaries with chosen/rejected pairs
        processor: Text processor/tokenizer
        config: Qwen3TTSConfig
    """
    
    def __init__(
        self,
        data_list: List[Dict],
        processor,
        config: Qwen3TTSConfig,
    ):
        self.data_list = data_list
        self.processor = processor
        self.config = config
    
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
        Get a single DPO sample.
        
        Returns:
            Dictionary with keys:
                - text_ids: Text token IDs
                - chosen_audio_codes: Audio codes for chosen response
                - rejected_audio_codes: Audio codes for rejected response
                - language: Language code
        """
        item = self.data_list[idx]
        
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        language = item.get('language', 'auto')
        
        # Build assistant text
        text = self._build_assistant_text(prompt)
        text_ids = self._tokenize_texts(text)
        
        # Load chosen audio codes
        if "audio_codes" in chosen:
            chosen_audio_codes = torch.tensor(chosen["audio_codes"], dtype=torch.long)
        else:
            raise ValueError("chosen audio_codes not found")
        
        # Load rejected audio codes
        if "audio_codes" in rejected:
            rejected_audio_codes = torch.tensor(rejected["audio_codes"], dtype=torch.long)
        else:
            raise ValueError("rejected audio_codes not found")
        
        return {
            "text_ids": text_ids,
            "chosen_audio_codes": chosen_audio_codes,
            "rejected_audio_codes": rejected_audio_codes,
            "language": language,
        }
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """
        Collate function for DPO DataLoader.
        
        Returns batch with chosen and rejected responses.
        """
        b = len(batch)
        
        # Calculate max lengths
        max_text_len = max(b['text_ids'].shape[1] for b in batch)
        max_chosen_len = max(b['chosen_audio_codes'].shape[0] for b in batch)
        max_rejected_len = max(b['rejected_audio_codes'].shape[0] for b in batch)
        max_codec_len = max(max_chosen_len, max_rejected_len)
        
        max_length = max_text_len + max_codec_len + 8
        
        # Initialize tensors for chosen
        chosen_input_ids = torch.zeros((b, max_length, 2), dtype=torch.long)
        chosen_codec_ids = torch.zeros((b, max_length, 16), dtype=torch.long)
        chosen_text_embedding_mask = torch.zeros((b, max_length), dtype=torch.bool)
        chosen_codec_embedding_mask = torch.zeros((b, max_length), dtype=torch.bool)
        chosen_codec_mask = torch.zeros((b, max_length), dtype=torch.bool)
        chosen_attention_mask = torch.zeros((b, max_length), dtype=torch.long)
        chosen_codec_0_labels = torch.full((b, max_length), -100, dtype=torch.long)
        
        # Same for rejected
        rejected_input_ids = torch.zeros((b, max_length, 2), dtype=torch.long)
        rejected_codec_ids = torch.zeros((b, max_length, 16), dtype=torch.long)
        rejected_text_embedding_mask = torch.zeros((b, max_length), dtype=torch.bool)
        rejected_codec_embedding_mask = torch.zeros((b, max_length), dtype=torch.bool)
        rejected_codec_mask = torch.zeros((b, max_length), dtype=torch.bool)
        rejected_attention_mask = torch.zeros((b, max_length), dtype=torch.long)
        rejected_codec_0_labels = torch.full((b, max_length), -100, dtype=torch.long)
        
        for i, data in enumerate(batch):
            text_ids = data['text_ids']
            chosen_audio_codes = data['chosen_audio_codes']
            rejected_audio_codes = data['rejected_audio_codes']
            
            text_ids_len = text_ids.shape[1]
            chosen_codec_len = chosen_audio_codes.shape[0]
            rejected_codec_len = rejected_audio_codes.shape[0]
            
            # Fill chosen
            self._fill_tensors(
                i, chosen_input_ids, chosen_codec_ids, chosen_text_embedding_mask,
                chosen_codec_embedding_mask, chosen_codec_mask, chosen_attention_mask,
                chosen_codec_0_labels, text_ids, chosen_audio_codes, text_ids_len,
                chosen_codec_len,
            )
            
            # Fill rejected
            self._fill_tensors(
                i, rejected_input_ids, rejected_codec_ids, rejected_text_embedding_mask,
                rejected_codec_embedding_mask, rejected_codec_mask, rejected_attention_mask,
                rejected_codec_0_labels, text_ids, rejected_audio_codes, text_ids_len,
                rejected_codec_len,
            )
        
        return {
            'chosen': {
                'input_ids': chosen_input_ids,
                'attention_mask': chosen_attention_mask,
                'text_embedding_mask': chosen_text_embedding_mask.unsqueeze(-1),
                'codec_embedding_mask': chosen_codec_embedding_mask.unsqueeze(-1),
                'codec_0_labels': chosen_codec_0_labels,
                'codec_ids': chosen_codec_ids,
                'codec_mask': chosen_codec_mask,
            },
            'rejected': {
                'input_ids': rejected_input_ids,
                'attention_mask': rejected_attention_mask,
                'text_embedding_mask': rejected_text_embedding_mask.unsqueeze(-1),
                'codec_embedding_mask': rejected_codec_embedding_mask.unsqueeze(-1),
                'codec_0_labels': rejected_codec_0_labels,
                'codec_ids': rejected_codec_ids,
                'codec_mask': rejected_codec_mask,
            },
        }
    
    def _fill_tensors(
        self,
        i: int,
        input_ids: torch.Tensor,
        codec_ids: torch.Tensor,
        text_embedding_mask: torch.Tensor,
        codec_embedding_mask: torch.Tensor,
        codec_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        codec_0_labels: torch.Tensor,
        text_ids: torch.Tensor,
        audio_codecs: torch.Tensor,
        text_ids_len: int,
        codec_ids_len: int,
    ):
        """Fill tensors for a single sample (chosen or rejected)."""
        audio_codec_0 = audio_codecs[:, 0]
        
        # Text channel
        input_ids[i, :3, 0] = text_ids[0, :3]
        input_ids[i, 3:7, 0] = self.config.tts_pad_token_id
        input_ids[i, 7, 0] = self.config.tts_bos_token_id
        input_ids[i, 8:8 + text_ids_len - 3, 0] = text_ids[0, 3:]
        input_ids[i, 8 + text_ids_len - 3, 0] = self.config.tts_eos_token_id
        input_ids[i, 8 + text_ids_len - 2:8 + text_ids_len + codec_ids_len, 0] = (
            self.config.tts_pad_token_id
        )
        text_embedding_mask[i, :8 + text_ids_len + codec_ids_len] = True
        
        # Codec channel (no-thinking pattern for DPO)
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
        input_ids[i, 8 + text_ids_len - 2, 1] = self.config.talker_config.codec_bos_id
        input_ids[i, 8 + text_ids_len - 1:8 + text_ids_len - 1 + codec_ids_len, 1] = audio_codec_0
        input_ids[i, 8 + text_ids_len - 1 + codec_ids_len, 1] = (
            self.config.talker_config.codec_eos_token_id
        )
        
        # Labels
        codec_0_labels[i, 8 + text_ids_len - 1:8 + text_ids_len - 1 + codec_ids_len] = audio_codec_0
        codec_0_labels[i, 8 + text_ids_len - 1 + codec_ids_len] = (
            self.config.talker_config.codec_eos_token_id
        )
        
        # All codebooks
        codec_ids[i, 8 + text_ids_len - 1:8 + text_ids_len - 1 + codec_ids_len, :] = audio_codecs
        
        # Masks
        codec_embedding_mask[i, 3:8 + text_ids_len + codec_ids_len] = True
        codec_embedding_mask[i, 3] = False  # for speaker embedding
        
        codec_mask[i, 8 + text_ids_len - 1:8 + text_ids_len - 1 + codec_ids_len] = True
        attention_mask[i, :8 + text_ids_len + codec_ids_len] = True


def load_dpo_dataset_from_jsonl(
    jsonl_path: str,
    processor,
    config: Qwen3TTSConfig,
) -> Qwen3TTSDPODataset:
    """
    Load DPO dataset from JSONL file.
    
    JSONL format:
    {
        "prompt": "text prompt",
        "chosen": {"audio": "chosen.wav", "audio_codes": [...]},
        "rejected": {"audio": "rejected.wav", "audio_codes": [...]},
        "language": "chinese"
    }
    
    Args:
        jsonl_path: Path to JSONL file
        processor: Text processor/tokenizer
        config: Qwen3TTSConfig
    
    Returns:
        Qwen3TTSDPODataset instance
    """
    data_list = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data_list.append(item)
    
    return Qwen3TTSDPODataset(
        data_list=data_list,
        processor=processor,
        config=config,
    )