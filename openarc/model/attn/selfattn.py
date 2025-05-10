import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

class SelfPatternAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if dim % heads != 0: raise ValueError("dim must be divisible by heads.")
        self.heads = heads; self.head_dim = dim // heads; self.scale = self.head_dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attention: bool = False, attention_mask: Optional[torch.Tensor] = None):
        B, N, D = x.shape
        if N == 0:
            empty_out = torch.zeros_like(x)
            empty_attn = torch.empty(B, self.heads, N, N, device=x.device, dtype=x.dtype) if return_attention else None
            return (empty_out, empty_attn) if return_attention else empty_out

        qkv = self.to_qkv(x).view(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_mask is not None: attn_scores = attn_scores + attention_mask # Assumes mask is broadcastable
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs_dropped = self.attn_drop(attn_probs)
        out = torch.matmul(attn_probs_dropped, v).transpose(1, 2).reshape(B, N, D)
        out_final = self.out_drop(self.proj(out))
        return (out_final, attn_probs) if return_attention else out_final
    