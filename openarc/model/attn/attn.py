
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import warnings
import math

from openarc.model.attn.crossattn import CrossPatternAttention
from openarc.model.attn.selfattn import SelfPatternAttention
from openarc.model.attn.rotary import RotaryEmbedding, Cache
from openarc.model.module.module import apply_rotary_pos_emb, RMSNorm

from openarc.config.config import config as C

class ARCAttention(nn.Module):
    def __init__(self, config=C, layer_idx: Optional[int] = None):
        super().__init__()
        config.__post_init__()
        self.config = config; self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size; self.num_heads = config.num_attention_heads
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim

        if self.qk_rope_head_dim > 0:
            self.rotary_emb = RotaryEmbedding(self.qk_rope_head_dim, config.max_position_embeddings, config.rope_theta)
        else: self.rotary_emb = None

        self.q_a_proj = nn.Linear(self.hidden_size, config.q_lora_rank, bias=config.attention_bias)
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)

        kv_lora_output_dim_for_rope = self.num_heads * self.qk_rope_head_dim if self.qk_rope_head_dim > 0 else 0
        self.kv_a_proj_with_mqa = nn.Linear(self.hidden_size, config.kv_lora_rank + kv_lora_output_dim_for_rope, bias=config.attention_bias)
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(config.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=config.attention_bias)
        
        self.num_self_pattern_modules = config.num_self_pattern_modules
        self.self_mpa = nn.ModuleList([SelfPatternAttention(dim=config.hidden_size, heads=config.pattern_heads, dropout=config.attention_dropout) for _ in range(self.num_self_pattern_modules)])
        self.num_cross_pattern_modules = config.num_cross_pattern_modules
        self.cross_mpa = nn.ModuleList([CrossPatternAttention(dim=config.hidden_size, heads=config.pattern_heads, dropout=config.attention_dropout) for _ in range(self.num_cross_pattern_modules)])
        
        self._gate_proj_paths = 1 + self.num_self_pattern_modules + (self.num_cross_pattern_modules if self.num_cross_pattern_modules > 0 else 0)
        self.gate_proj = nn.Linear(self.hidden_size, self._gate_proj_paths)
        self.softmax_scale = self.q_head_dim**-0.5 if self.q_head_dim > 0 else 1.0
        
        self.num_attention_cycles = config.num_attention_cycles
        
        if self.num_attention_cycles > 1:
            self.cycle_layernorms = nn.ModuleList([RMSNorm(self.hidden_size, eps=config.rms_norm_eps) for _ in range(self.num_attention_cycles - 1)])
        else: self.cycle_layernorms = None

    def _prepare_attention_mask(self, attention_mask: Optional[torch.Tensor], target_q_len: int, target_kv_len: int, bsz: int, num_heads: int) -> Optional[torch.Tensor]:
        if attention_mask is None: return None
        # Ensure mask is 4D: (bsz or 1, num_heads or 1, q_len, kv_len)
        if attention_mask.ndim == 2: # (q_len, kv_len) -> (1, 1, q_len, kv_len)
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        elif attention_mask.ndim == 3: # (bsz, q_len, kv_len) -> (bsz, 1, q_len, kv_len)
            attention_mask = attention_mask.unsqueeze(1)
        
        # Final check and broadcast/slice if necessary
        if attention_mask.shape[0] != bsz and attention_mask.shape[0] != 1:
            raise ValueError(f"Mask batch size {attention_mask.shape[0]} incompatible with input batch size {bsz}.")
        if attention_mask.shape[1] != num_heads and attention_mask.shape[1] != 1:
             raise ValueError(f"Mask head dim {attention_mask.shape[1]} incompatible with num_heads {num_heads}.")

        # Slice mask if it's larger than target dimensions
        mask_q, mask_k = attention_mask.shape[-2], attention_mask.shape[-1]
        if mask_q > target_q_len: attention_mask = attention_mask[..., :target_q_len, :]
        if mask_k > target_kv_len: attention_mask = attention_mask[..., :, :target_kv_len]
        
        if attention_mask.shape[-2] != target_q_len or attention_mask.shape[-1] != target_kv_len:
            # This implies the mask provided was smaller or an intermediate logic error.
            # This is a common place for issues. For now, we rely on the caller providing a compatible mask
            # or one that can be sliced down. If it's smaller, it's problematic.
            warnings.warn(f"Attention mask final shape {attention_mask.shape} does not match target ({target_q_len}, {target_kv_len}). Potential errors.")
        return attention_mask


    def _single_attention_pass(
        self, current_hidden_states: torch.Tensor, initial_kv_source: torch.Tensor,
        context_states: Optional[torch.Tensor], attention_mask_main: Optional[torch.Tensor],
        cross_attention_mask: Optional[torch.Tensor], position_ids: Optional[torch.LongTensor],
        current_pass_past_key_value: Optional[Cache], current_pass_use_cache: bool,
        current_pass_output_attentions: bool, current_pass_layer_idx: Optional[int]
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:

        bsz, q_len, _ = current_hidden_states.shape

        q_mid = self.q_a_layernorm(self.q_a_proj(current_hidden_states))
        q_proj = self.q_b_proj(q_mid).view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = (torch.split(q_proj, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1) if self.qk_rope_head_dim > 0 else (q_proj, None))

        source_for_kv = initial_kv_source
        if self.qk_rope_head_dim > 0:
            kv_lora_output_dim_for_rope = self.num_heads * self.qk_rope_head_dim
            split_sizes_kv_a = [self.config.kv_lora_rank, kv_lora_output_dim_for_rope]
            compressed_kv_parts = self.kv_a_proj_with_mqa(source_for_kv)
            kv_lora_part, k_pe_all_heads_flat = torch.split(compressed_kv_parts, split_sizes_kv_a, dim=-1)
            k_pe_all_heads_new = k_pe_all_heads_flat.view(bsz, q_len, self.num_heads, self.qk_rope_head_dim).transpose(1,2) # q_len is len of new keys
        else:
            kv_lora_part = self.kv_a_proj_with_mqa(source_for_kv)
            k_pe_all_heads_new = None
        
        kv_mid = self.kv_a_layernorm(kv_lora_part)
        kv_proj = self.kv_b_proj(kv_mid).view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1,2)
        k_nope_new, v_states_new = torch.split(kv_proj, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        query_states_final = q_nope
        key_states_for_concat = k_nope_new

        if self.rotary_emb and q_pe is not None and k_pe_all_heads_new is not None:
            max_pos_for_rope = q_len
            if position_ids is not None and position_ids.numel() > 0: max_pos_for_rope = max(q_len, position_ids.max().item() + 1)
            if max_pos_for_rope > 0:
                cos, sin = self.rotary_emb(x_tensor_for_meta=q_pe, seq_len=max_pos_for_rope)
                q_rot, k_rot_new_pe = apply_rotary_pos_emb(q_pe, k_pe_all_heads_new, cos, sin, position_ids)
                query_states_final = torch.cat((q_nope, q_rot), dim=-1) if q_nope is not None and q_nope.numel() > 0 else q_rot
                key_states_for_concat = torch.cat((k_nope_new, k_rot_new_pe), dim=-1) if k_nope_new is not None and k_nope_new.numel() > 0 else k_rot_new_pe
        
        key_states_main_attn = key_states_for_concat
        v_states_main_attn = v_states_new
        len_new_keys = key_states_main_attn.shape[2]

        if current_pass_past_key_value is not None and current_pass_layer_idx is not None:
            cached_kv = current_pass_past_key_value.get_kv(current_pass_layer_idx)
            if cached_kv is not None:
                past_k, past_v = cached_kv
                key_states_main_attn = torch.cat([past_k.to(key_states_main_attn.device), key_states_main_attn], dim=2)
                v_states_main_attn = torch.cat([past_v.to(v_states_main_attn.device), v_states_main_attn], dim=2)
            if current_pass_use_cache:
                current_pass_past_key_value.update(key_states_main_attn, v_states_main_attn, current_pass_layer_idx)
        
        kv_seq_len_main_attn = key_states_main_attn.shape[-2]
        attn_weights = torch.matmul(query_states_final, key_states_main_attn.transpose(2, 3)) * self.softmax_scale
        
        prepared_mask_main = self._prepare_attention_mask(attention_mask_main, q_len, kv_seq_len_main_attn, bsz, self.num_heads)
        if prepared_mask_main is not None: attn_weights = attn_weights + prepared_mask_main

        attn_probs_main = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states_final.dtype)
        attn_probs_main_dropped = F.dropout(attn_probs_main, p=self.config.attention_dropout, training=self.training)
        attn_output_main = torch.matmul(attn_probs_main_dropped, v_states_main_attn)
        attn_output_main = self.o_proj(attn_output_main.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.v_head_dim))

        all_outputs_to_gate = [attn_output_main]
        all_attn_probs_for_return: Dict[str, Any] = {"main": attn_probs_main if current_pass_output_attentions else None}

        self_mpa_mask = self._prepare_attention_mask(attention_mask_main, q_len, q_len, bsz, self.config.pattern_heads) # Self-attn mask is q_len x q_len
        for i, pattern_module in enumerate(self.self_mpa):
            p_out, p_attn = pattern_module(current_hidden_states, return_attention=True, attention_mask=self_mpa_mask)
            all_outputs_to_gate.append(p_out)
            if current_pass_output_attentions: all_attn_probs_for_return[f"self_pattern_{i}"] = p_attn
        
        num_active_cross_paths = 0
        if self.cross_mpa and context_states is not None:
            num_active_cross_paths = len(self.cross_mpa)
            cross_mask_prepared = self._prepare_attention_mask(cross_attention_mask, q_len, context_states.shape[1], bsz, self.config.pattern_heads)
            for i, cross_module in enumerate(self.cross_mpa):
                cp_out, cp_attn = cross_module(current_hidden_states, context_states, return_attention=True, attention_mask=cross_mask_prepared)
                all_outputs_to_gate.append(cp_out)
                if current_pass_output_attentions: all_attn_probs_for_return[f"cross_pattern_{i}"] = cp_attn
        
        expected_gate_dim = 1 + self.num_self_pattern_modules + num_active_cross_paths
        gate_values_raw = self.gate_proj(current_hidden_states) # Gate based on current iteration's hidden states
        gate_values = F.softmax(gate_values_raw[..., :expected_gate_dim], dim=-1)

        combined_output = torch.zeros_like(attn_output_main)
        if len(all_outputs_to_gate) != expected_gate_dim:
            raise RuntimeError(f"Internal Mismatch: {len(all_outputs_to_gate)} outputs for gating, but gate expects {expected_gate_dim}.")
        for i, out_tensor in enumerate(all_outputs_to_gate):
            combined_output = combined_output + out_tensor * gate_values[..., i].unsqueeze(-1)
            
        final_attn_probs = all_attn_probs_for_return if current_pass_output_attentions and any(v is not None for v in all_attn_probs_for_return.values()) else None
        return combined_output, final_attn_probs

    def forward(
        self, hidden_states: torch.Tensor, context_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None, cross_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None, past_key_value: Optional[Cache] = None,
        output_attentions: bool = False, use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]], Optional[Cache]]:

        current_cycle_hs = hidden_states
        final_pass_attentions: Optional[Dict[str, Any]] = None

        for cycle_idx in range(self.num_attention_cycles):
            if cycle_idx > 0 and self.cycle_layernorms:
                current_cycle_hs = self.cycle_layernorms[cycle_idx - 1](current_cycle_hs)

            is_first_cycle = cycle_idx == 0
            is_last_cycle = cycle_idx == self.num_attention_cycles - 1
            
            # KV cache is updated by the first cycle based on the original hidden_states.
            # Subsequent cycles use the Q from refined HS but attend to the K/V established by the first cycle (and its cache).
            pass_use_cache_update = use_cache and is_first_cycle
            pass_output_attns_for_cycle = output_attentions and is_last_cycle

            attn_output_this_cycle, attn_probs_this_cycle = self._single_attention_pass(
                current_hidden_states=current_cycle_hs,
                initial_kv_source=hidden_states, # K/V for main path from original HS
                context_states=context_states,
                attention_mask_main=attention_mask,
                cross_attention_mask=cross_attention_mask,
                position_ids=position_ids,
                current_pass_past_key_value=past_key_value,
                current_pass_use_cache=pass_use_cache_update,
                current_pass_output_attentions=pass_output_attns_for_cycle,
                current_pass_layer_idx=self.layer_idx
            )
            
            # Residual connection for the cycle
            current_cycle_hs = current_cycle_hs + attn_output_this_cycle

            if is_last_cycle:
                final_pass_attentions = attn_probs_this_cycle
        
        return current_cycle_hs, final_pass_attentions, past_key_value

