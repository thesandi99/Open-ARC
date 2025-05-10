import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class CrossPatternAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if dim % heads != 0: raise ValueError("dim must be divisible by heads.")
        self.heads = heads; self.head_dim = dim // heads; self.scale = self.head_dim ** -0.5
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x_query: torch.Tensor, x_context: torch.Tensor, return_attention: bool = False, attention_mask: Optional[torch.Tensor] = None):
        B_q, N_q, D_q = x_query.shape; B_c, N_c, D_c = x_context.shape
        if N_q == 0 or N_c == 0:
            empty_out = torch.zeros_like(x_query)
            empty_attn = torch.empty(B_q, self.heads, N_q, N_c, device=x_query.device, dtype=x_query.dtype) if return_attention else None
            return (empty_out, empty_attn) if return_attention else empty_out
        
        q = self.to_q(x_query).view(B_q, N_q, self.heads, self.head_dim).permute(0, 2, 1, 3)
        kv_ctx = self.to_kv(x_context).view(B_c, N_c, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k_ctx, v_ctx = kv_ctx[0], kv_ctx[1]
        attn_scores = torch.matmul(q, k_ctx.transpose(-2, -1)) * self.scale
        if attention_mask is not None: attn_scores = attn_scores + attention_mask # Assumes mask is broadcastable
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs_dropped = self.attn_drop(attn_probs)
        out = torch.matmul(attn_probs_dropped, v_ctx).transpose(1, 2).reshape(B_q, N_q, D_q)
        out_final = self.out_drop(self.proj(out))
        return (out_final, attn_probs) if return_attention else out_final
    