
import torch
import torch.nn as nn
import torch.nn.functional as F

from openarc.model.module.module import RMSNorm
from openarc.config.config import config as C

# after the BCHead the decoder started and in decoder the mlp is used
# this is the mlp module but the normal module of mlp is kind of liner nn module with 2 linear layers maybe 3 
# but in this i want to add the 48 expert to connect and talk each other to get the final output

class _ExpertMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, activation_fn: nn.Module, dropout_rate: float):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.activation_fn = activation_fn
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

    
class MultiScaleFeatureExtractor(nn.Module):
    """
    Applies multiple 1D convolutions with different kernel sizes in parallel
    and concatenates their outputs. Preserves sequence length.
    Input: (batch_size, in_channels, sequence_length)
    Output: (batch_size, total_out_channels, sequence_length)
    """
    def __init__(self, in_channels: int, per_branch_channels: int, kernel_sizes=[1, 3, 5], activation_fn=nn.ReLU(inplace=True)):
        super().__init__()
        self.conv_branches = nn.ModuleList()
        total_out_channels = 0
        for ks in kernel_sizes:
            padding = ks // 2 # Keeps sequence length same for odd kernels
            self.conv_branches.append(
                nn.Conv1d(in_channels, per_branch_channels, kernel_size=ks, padding=padding)
            )
            total_out_channels += per_branch_channels
        self.activation_fn = activation_fn
        self.output_channels = total_out_channels # The channel dim after concatenation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x input shape: (batch_size, in_channels, sequence_length)
        branch_outputs = [self.activation_fn(conv_branch(x)) for conv_branch in self.conv_branches]
        # Concatenate along the channel dimension
        
        x_concat = torch.cat(branch_outputs, dim=1)
        # Output shape: (batch_size, self.output_channels, sequence_length)
        return x_concat
    
class TExpert(nn.Module): 
    def __init__(self, config=C):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size # Input hidden_size to TExpert
        self.num_experts = config.num_experts

        self.num_refinement_steps = config.expert_refinement_steps

        if self.num_refinement_steps < 1:
            raise ValueError("num_refinement_steps must be at least 1.")


        expert_mlp_intermediate_size = config.mlp_intermediate_size
        mlp_activation_str = config.mlp_activation
        if mlp_activation_str == "silu": expert_activation_fn = nn.SiLU()
        elif mlp_activation_str == "gelu": expert_activation_fn = nn.GELU()
        elif mlp_activation_str == "relu": expert_activation_fn = nn.ReLU()
        else: raise ValueError(f"Unsupported activation: {mlp_activation_str}")
        expert_dropout_rate = getattr(config, 'dropout_rate', 0.1)

        self.use_feature_extractor = config.use_moe_feature_extractor
        if self.use_feature_extractor:

            num_conv_kernels = len(config.moe_feature_extractor_kernel_sizes)
            per_branch_channels = self.hidden_size // num_conv_kernels
            if per_branch_channels <=0: per_branch_channels = self.hidden_size # fallback if too small

            self.feature_extractor = MultiScaleFeatureExtractor(
                in_channels=self.hidden_size, # Takes original hidden_size
                per_branch_channels=per_branch_channels,
                kernel_sizes=config.moe_feature_extractor_kernel_sizes
            )

            # The dimension for experts and gate will be the output of the feature extractor
            self.dim_for_experts_and_gate = self.feature_extractor.output_channels
            
            if self.dim_for_experts_and_gate != self.hidden_size:
                self.final_projection = nn.Linear(self.dim_for_experts_and_gate, self.hidden_size)
            else:
                self.final_projection = nn.Identity()
        else:
            self.feature_extractor = None
            self.dim_for_experts_and_gate = self.hidden_size
            self.final_projection = nn.Identity()

        self.gate = nn.Linear(self.dim_for_experts_and_gate, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [_ExpertMLP(self.dim_for_experts_and_gate,
                        expert_mlp_intermediate_size, 
                        expert_activation_fn,
                        expert_dropout_rate)
             for _ in range(self.num_experts)]
        )

        if self.num_refinement_steps > 1:
            self.iter_layernorms = nn.ModuleList([
                RMSNorm(self.dim_for_experts_and_gate, eps=getattr(config, 'rms_norm_eps', 1e-6))
                for _ in range(self.num_refinement_steps -1) # One LN before each refinement step > 0
            ]) if getattr(config, 'rms_norm_eps', 1e-6) > 0 else None
        else:
            self.iter_layernorms = None


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (batch_size, sequence_length, self.hidden_size)
        batch_size, sequence_length, _ = hidden_states.shape

        # 1. Optional: Apply MultiScaleFeatureExtractor ONCE at the beginning
        if self.feature_extractor is not None:
            # Permute for Conv1d: (B, S, H) -> (B, H, S)
            permuted_hidden_states = hidden_states.permute(0, 2, 1)
            extracted_features = self.feature_extractor(permuted_hidden_states) # (B, C_out, S)
            # Permute back: (B, C_out, S) -> (B, S, C_out)
            current_processing_states = extracted_features.permute(0, 2, 1)
            # current_processing_states now has shape (B, S, self.dim_for_experts_and_gate)
        else:
            current_processing_states = hidden_states # (B, S, self.hidden_size)
        
        # The input to the first residual connection of the loop
        # This should match the dimension of current_processing_states
        # loop_input_residual = current_processing_states # Not directly used like this, residual is additive within loop

        for i in range(self.num_refinement_steps):
            # Input for this step's gating and experts
            input_to_step = current_processing_states # This is the state from previous iteration or initial input

            if i > 0 and self.iter_layernorms is not None and self.iter_layernorms[i-1] is not None:
                # Normalize the output of the previous step before feeding to the current step
                input_to_step = self.iter_layernorms[i-1](current_processing_states) # Use current_processing_states which is previous output

            # Reshape for token-wise gating and expert processing
            # (B, S, dim_for_experts) -> (B*S, dim_for_experts)
            reshaped_input_to_step = input_to_step.reshape(-1, self.dim_for_experts_and_gate) # <--- FIX HERE

            router_logits = self.gate(reshaped_input_to_step)
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32).to(hidden_states.dtype)

            all_experts_output_list = []
            for expert in self.experts:
                all_experts_output_list.append(expert(reshaped_input_to_step))
            
            stacked_all_experts_output = torch.stack(all_experts_output_list, dim=1) # (B*S, num_experts, dim_for_experts)
            
            # Weighted sum
            weighted_expert_outputs = torch.bmm(routing_weights.unsqueeze(1), stacked_all_experts_output).squeeze(1) # (B*S, dim_for_experts)
            
            # Reshape back to (B, S, dim_for_experts)
            # Use .reshape() here too for safety, although it might not be strictly necessary
            # if weighted_expert_outputs is contiguous after bmm and squeeze.
            weighted_expert_outputs_reshaped = weighted_expert_outputs.reshape(batch_size, sequence_length, self.dim_for_experts_and_gate)

            # Residual connection within the loop
            # Add the output of this step's experts to the state *before* it was potentially normed for this step.
            # `current_processing_states` is the result of the PREVIOUS iteration.
            current_processing_states = current_processing_states + weighted_expert_outputs_reshaped


        # 3. Final Projection if dimensions changed due to feature_extractor
        final_output = self.final_projection(current_processing_states) # (B, S, self.hidden_size)
        
        return final_output
    
class MultiScaleConvAggregator(nn.Module):
    def __init__(self, in_channels: int, target_output_channels: int,
                 branch_channels_ratio: float = 0.25, min_branch_channels: int = 32,
                 kernel_sizes=[1, 3, 5], activation_fn=nn.ReLU(inplace=True)):
        super().__init__()
        self.in_channels = in_channels
        self.target_output_channels = target_output_channels

        self.conv_branches = nn.ModuleList()
        # Ensure calculated_branch_channels is at least 1
        calculated_branch_channels = max(1, min_branch_channels, int(target_output_channels * branch_channels_ratio))


        for ks in kernel_sizes:
            padding = ks // 2
            self.conv_branches.append(
                nn.Conv1d(in_channels, calculated_branch_channels, kernel_size=ks, padding=padding)
            )
        
        self.num_branches = len(kernel_sizes)
        self.activation_fn = activation_fn
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        
        combined_branch_channels = calculated_branch_channels * self.num_branches
        self.projection = nn.Linear(combined_branch_channels, self.target_output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x input shape: (batch_size, in_channels, sequence_length)
        branch_outputs = [self.activation_fn(conv_branch(x)) for conv_branch in self.conv_branches]

        x_concat = torch.cat(branch_outputs, dim=1)
        pooled = self.pool(x_concat)
        flattened = self.flatten(pooled)
        
        # Apply activation *after* projection usually, unless it's part of the MLP block inside projection
        output = self.activation_fn(self.projection(flattened)) # Or just self.projection(flattened) if activation is in EHead
        return output

