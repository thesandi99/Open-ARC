import torch
import torch.nn as nn
from typing import Optional, Tuple, List


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq_device = device if device is not None else (torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=inv_freq_device).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached: int = 0
        self.cos_cached: Optional[torch.Tensor] = None
        self.sin_cached: Optional[torch.Tensor] = None
        self._set_cos_sin_cache(seq_len=self.max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        self.max_seq_len_cached = seq_len
        if seq_len == 0:
            self.cos_cached = torch.empty(0, self.dim // 2, device=device, dtype=dtype) # Match freqs shape
            self.sin_cached = torch.empty(0, self.dim // 2, device=device, dtype=dtype)
            return
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(t.device)) # seq_len, dim / 2
        emb = torch.cat((freqs, freqs), dim=-1) # seq_len, dim
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)

    def forward(self, x_tensor_for_meta: Optional[torch.Tensor] = None, seq_len: Optional[int] = None):
        if seq_len is None:
            if x_tensor_for_meta is not None: seq_len = x_tensor_for_meta.shape[-2]
            else: raise ValueError("seq_len must be provided if x_tensor_for_meta is None.")
        
        _device = x_tensor_for_meta.device if x_tensor_for_meta is not None else self.inv_freq.device
        _dtype = x_tensor_for_meta.dtype if x_tensor_for_meta is not None else torch.get_default_dtype() # Use default if not inferable

        if self.cos_cached is None or seq_len > self.max_seq_len_cached or \
           self.cos_cached.device != _device or self.cos_cached.dtype != _dtype:
            self._set_cos_sin_cache(seq_len=max(seq_len, self.max_position_embeddings), device=_device, dtype=_dtype) # Cache at least max_pos_embeddings

        if self.cos_cached is None or self.sin_cached is None:
            raise RuntimeError("Rotary embedding cache not initialized.")
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]
    
class Cache:
    def __init__(self):
        self.past_key_values: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []

    def get_kv(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if 0 <= layer_idx < len(self.past_key_values) and self.past_key_values[layer_idx] is not None:
            return self.past_key_values[layer_idx]
        return None

    def update(self, key: torch.Tensor, value: torch.Tensor, layer_idx: int):
        if layer_idx < 0: raise IndexError("layer_idx cannot be negative.")
        while len(self.past_key_values) <= layer_idx: self.past_key_values.append(None)
        self.past_key_values[layer_idx] = (key.detach(), value.detach()) # Detach to prevent further graph tracking

    def reset(self): self.past_key_values = []