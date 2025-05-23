
# openarc/config/tiny.py

# Copyright (c) Sangram.
# Licensed under the MIT license.

import torch.nn.functional as F

# Model: OpenArc-Tiny
# Tokoenizer: "OpenARC-FlatGrid"
# Total number of modules/layers: 586
# Number of Linear layers: 247
# 80-epoch loss-0.0004 .pth 

# attention heads: 4 [4/2/2]
# task experts: 18 
# attention cycles: 10

class Config:
    def __init__(self, **kwargs):
        
        # Token IDs
        self.bos: int = 10
        self.grid_start_token: int = 11
        self.grid_end_token: int = 12
        self.train_ctx: int = 13
        self.test_ctx: int = 14
        self.pad: int = 15
        self.pred_ex_token: int = 16
        self.eos: int = 17
        self.input_grid_token: int = 18
        self.output_grid_token: int = 19

        # Vocab settings
        self.vocab_base_size = 10
        self.num_special_tokens = 10
        self.vocab_size: int = self.vocab_base_size + self.num_special_tokens

        # Compatibility
        self.use_pred_ex_format: bool = True
        self.ex_start = self.grid_start_token
        self.ex_end = self.grid_end_token
        self.input_grid = self.input_grid_token
        self.output_grid = self.output_grid_token

        # Model architecture
        self.hidden_size: int = 256
        self.max_position_embeddings: int = 8192  # typically power of 2

        # Decoder layers
        self.rms_norm_eps: float = 1e-6
        self.num_hidden_layers: int = 4
        self.tehead_num_layers: int = 2

        # Head
        self.intermediate_size = self.hidden_size * 2
        self.tehead_mlp_intermediate_size: int = self.hidden_size * 2
        self.tehead_mlp_activation: str = "gelu"

        self.active_summary_head: bool = True
        self.hidden_dropout_prob: float = 0.1
        self.attention_dropout: float = 0.1

        # Attention
        self.num_attention_heads: int = 4
        self.qk_rope_head_dim: int = 32
        self.qk_nope_head_dim: int = 32
        self.v_head_dim: int = self.hidden_size // self.num_attention_heads  # 64
        self.rope_theta: float = 10000.0
        self.q_lora_rank: int = 64
        self.kv_lora_rank: int = 64
        self.attention_bias: bool = False

        # Pattern attention modules
        self.pattern_heads: int = 4
        self.num_self_pattern_modules: int = 2
        self.num_cross_pattern_modules: int = 2
        self.num_attention_cycles: int = 10

        self._attn_implementation: str = "normal"

        # JModule
        self.jmodule_recurrent_layers: int = 3
        self.jmodule_input_size: int = self.hidden_size
        self.jmodule_hidden_size: int = self.hidden_size
        self.candidate_activation = F.relu
        self.hidden_activation = F.tanh
        self.jcell_candidate_activation = F.tanh
        self.jcell_hidden_activation = F.tanh
        self.layer_norm_eps: float = 1e-6

        # Mixture of Experts
        self.num_experts: int = 18
        self.expert_refinement_steps: int = 1
        self.mlp_intermediate_size = self.hidden_size * 4
        self.mlp_activation: str = 'silu'
        self.moe_feature_extractor_kernel_sizes = [3,5,7]
        self.moe_feature_extractor_per_branch_channels = self.hidden_size // len(self.moe_feature_extractor_kernel_sizes)
        self.use_moe_feature_extractor = True
        self.n_routed_experts: int = 4
        self.num_experts_per_tok: int = 1

        # Output head
        self.output_size: int = self.vocab_size

        # Training
        self.batch_size: int = 16
        self.num_epochs: int = 20
        self.learning_rate: float = 3e-4
        self.weight_decay: float = 0.01
        self.gradient_clip_val: float = 1.0
        self.use_amp: bool = True

        # Load user overrides
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Config trying to set unknown attribute {key}")

        self.__post_init__()
    
    # No Need To Change !
    def __post_init__(self):
        # Validate model shape compatibility
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        expected_v_head_dim = self.hidden_size // self.num_attention_heads
        if self.v_head_dim != expected_v_head_dim:
            print(f"Warning: v_head_dim ({self.v_head_dim}) != hidden_size/num_attention_heads ({expected_v_head_dim})")

        # Validate QK head dimensions
        total_qk_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
        if self.num_attention_heads * total_qk_dim != self.hidden_size and total_qk_dim > 0:
            pass  # acceptable depending on implementation
        elif total_qk_dim == 0:
            print("Warning: qk_rope_head_dim and qk_nope_head_dim are both 0. Setting default qk_nope_head_dim.")
            self.qk_nope_head_dim = self.hidden_size // self.num_attention_heads

        if self.n_routed_experts > 0 and self.num_experts_per_tok > self.n_routed_experts:
            print(f"Warning: num_experts_per_tok ({self.num_experts_per_tok}) > n_routed_experts ({self.n_routed_experts}).")
            self.num_experts_per_tok = self.n_routed_experts

config = Config()
