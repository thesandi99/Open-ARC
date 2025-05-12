
# [BOS, TRAIN_CTX, EX_START, IN_GRID, 1,1,1,1, OUT_GRID, 0,0,1,1, EX_END,  (next ex...)  TEST_CTX, EX_START, IN_GRID, 3,3,3,3, OUT_GRID, EOS]
# 16, 13, 10, 11, 1,1,1,1, 12, 0,0,1,1, 10, 14, 10, 11, 3,3,3,3, 12, 17

import torch.nn.functional as F

class Config:
    def __init__(self, **kwargs):
        
        # special tokens
        self.pad: int = 15

        # begin & end of sequence
        self.bos: int = 16
        self.eos: int = 16

        # train & test
        self.train_ctx: int = 13
        self.test_ctx: int = 14
        

        # example start and end 
        self.ex_start: int = 10
        self.ex_end: int = 10

        # input and output pair
        self.input_grid: int = 11
        self.output_grid: int = 12
        self.vocab_size: int = 10 + 7 
        
        # ------------------

        # Model Architecture
        # OpenEmbedder
        self.hidden_size: int = 512
        self.max_position_embeddings: int = 4096 * 8 # 32768
        
        # OpenDecoder
        self.rms_norm_eps: float = 1e-6
        self.num_hidden_layers: int = 2 # decoderLayer
        self.tehead_num_layers: int = 2 # TEHead

        # head 
        self.intermediate_size = self.hidden_size // 2
        self.tehead_mlp_intermediate_size: int = self.hidden_size // 2
        self.tehead_mlp_activation: str = "gelu"

        self.active_summary_head: bool = False
        self.hidden_dropout_prob: float = 0.1
        self.attention_dropout: float = 0.1

        # Attention Core
        self.num_attention_heads: int = 8
        
        self.qk_nope_head_dim: int = 32
        self.qk_rope_head_dim: int = 32
        
        self.v_head_dim: int = 64 # q_head_dim * num_heads should be divisible by hidden_size for o_proj
        self.rope_theta: float = 10000.0
        
        self.q_lora_rank: int = 128
        self.kv_lora_rank: int = 128
        self.attention_bias: bool = False

        self.pattern_heads: int = 4
        self.num_self_pattern_modules: int = 1
        self.num_cross_pattern_modules: int = 1
        self.num_attention_cycles: int = 1 # For iterative attention

        # Internal / Attention Implementation choice
        self._attn_implementation: str = "normal" 

        # JModule 
        self.jmodule_recurrent_layers: int = 3 # jcell  layer 3 recommended
        self.jmodule_input_size: int  = 512
        self.jmodule_hidden_size: int = 512
        self.candidate_activation = F.relu
        self.hidden_activation = F.tanh
        self.jcell_candidate_activation = F.tanh
        self.jcell_hidden_activation = F.tanh

        self.layer_norm_eps: float = 1e-6
        
        # Task Experts
        self.num_experts: int = 12
        self.expert_refinement_steps: int = 2
        self.mlp_intermediate_size = self.hidden_size * 4
        self.mlp_activation: str = 'silu'

        self.moe_feature_extractor_kernel_sizes = [1, 3, 5]
        self.moe_feature_extractor_per_branch_channels = self.hidden_size // len(self.moe_feature_extractor_kernel_sizes)
        self.use_moe_feature_extractor = True
        self.n_routed_experts: int = 10
        self.num_experts_per_tok: int = 1
        
        # lm_head

        self.output_size: int = self.vocab_size
        
        # No Need To Change !
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Config trying to set unknown attribute {key}")
        
        self.__post_init__()


    def __post_init__(self):
        # Validate and derive some values
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        
        # Standard head dimension for V if not specified or inconsistent
        calculated_v_head_dim = self.hidden_size // self.num_attention_heads
        if self.v_head_dim != calculated_v_head_dim:
            print(f"Warning: Config v_head_dim ({self.v_head_dim}) differs from calculated "
                  f"hidden_size/num_attention_heads ({calculated_v_head_dim}). Using configured v_head_dim.")

        # Total QK head dimension based on RoPE/NoPE parts
        total_qk_head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
        if self.num_attention_heads * total_qk_head_dim != self.hidden_size and total_qk_head_dim > 0:
             pass # This is fine with current ARCAttention structure.
        elif total_qk_head_dim == 0 and self.num_attention_heads > 0:
            # If no RoPE and no NoPE, QK head dim is 0, which is problematic.
            # Default to standard head dim if both are zero.
            print("Warning: qk_rope_head_dim and qk_nope_head_dim are both 0. Defaulting qk_nope_head_dim.")
            self.qk_nope_head_dim = self.hidden_size // self.num_attention_heads


        if self.n_routed_experts > 0 and self.num_experts_per_tok > self.n_routed_experts:
            print(f"Warning: num_experts_per_tok ({self.num_experts_per_tok}) > n_routed_experts ({self.n_routed_experts}). "
                  f"Setting num_experts_per_tok to {self.n_routed_experts}.")
            self.num_experts_per_tok = self.n_routed_experts


from openarc.config.nanomodel import config
config = config