import torch
import torch.nn as nn

from openarc.model.module.module import RMSNorm
from openarc.model.module.expert import TExpert
from openarc.model.module.head import TEHead
from openarc.config.config import config as C

from openarc.model.attn.attn import ARCAttention

#  TODO: 
ATTENTION_CLASSES = {
    "normal": ARCAttention,
}


class _OpenDecoderLayer(nn.Module): 
    def __init__(self, config=C, layer_idx: int = 0 ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx 

        # Layer Normalization before Self-Attention
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Self-Attention Mechanism
        if config._attn_implementation not in ATTENTION_CLASSES:
            raise ValueError(
                f"Attention implementation '{config._attn_implementation}' not found. "
                f"Available: {list(ATTENTION_CLASSES.keys())}"
            )
        
        self._attn = ATTENTION_CLASSES[config._attn_implementation](
            config=config, layer_idx=layer_idx
        )
        # Dropout after attention, if specified (e.g., config.hidden_dropout_prob)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Layer Normalization before the FFN (TExpert)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Feed-Forward Network (using TExpert)
        self.mlp = TExpert(config) # TExpert is your MoE FFN

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None, # Causal mask for decoders
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        # 1. Self-Attention Block (with Pre-LN)
        residual = hidden_states
        normed_hidden_states = self.input_layernorm(hidden_states)

        # Attention mechanism
        attn_outputs = self._attn(
            normed_hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        attn_output = attn_outputs[0] # The actual attention output
        present_key_value = attn_outputs[1] if use_cache else None
        raw_attn_weights = attn_outputs[2] if output_attentions else None


        # Apply dropout if configured (common after attention projections)
        attn_output = self.dropout(attn_output)
        hidden_states = residual + attn_output # Add residual

        # 2. Feed-Forward Network Block (TExpert, with Pre-LN)
        residual = hidden_states
        normed_hidden_states = self.post_attention_layernorm(hidden_states)
        
        ffn_output = self.mlp(normed_hidden_states) # Pass through TExpert
        
        # Apply dropout if configured (common after FFN)
        ffn_output = self.dropout(ffn_output) # Using the same dropout instance, or have a separate one
        hidden_states = residual + ffn_output # Add residual

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (raw_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
            
        return outputs # (hidden_states, present_key_value, attentions)


class OpenDecoder(nn.Module):
    def __init__(self, config=C):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [_OpenDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # Final LayerNorm (often applied after the last decoder layer)
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if config.active_summary_head:
            mlp_cfg_tehead_dict = {
                'intermediate_size': config.tehead_mlp_intermediate_size,
                'mlp_activation': config.tehead_mlp_activation,
            }
            mlp_config_for_tehead = C(**mlp_cfg_tehead_dict)
            
            self.summary_head = TEHead(
                dim=config.hidden_size, # Takes final hidden_size
                mlp_config=mlp_config_for_tehead,
                num_layer=config.tehead_num_layers,
            )
        else:
            self.summary_head = None

    def forward(self, hidden_states, attention_mask=None, past_key_values=None, use_cache=None, output_attentions=None):
        all_hidden_states = () if getattr(self.config, 'output_hidden_states_all', False) else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for i, layer_module in enumerate(self.layers):
            if all_hidden_states is not None:
                all_hidden_states += (hidden_states,)

            layer_past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=layer_past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[1],) # present_key_value
            if output_attentions:
                all_self_attns += (layer_outputs[2] if len(layer_outputs) > 2 else layer_outputs[1],)


        hidden_states = self.final_layernorm(hidden_states)

        if all_hidden_states is not None:
            all_hidden_states += (hidden_states,)
            
        sequence_summary = None
        if self.summary_head is not None:
            sequence_summary = self.summary_head(hidden_states) # Pass the final processed hidden_states

        # Standard Transformer output structure
        output_tuple = (hidden_states,)
        if use_cache: output_tuple += (next_decoder_cache,)
        if all_hidden_states is not None: output_tuple += (all_hidden_states,)
        if all_self_attns is not None: output_tuple += (all_self_attns,)
        if sequence_summary is not None: output_tuple += (sequence_summary,)
        
        if len(output_tuple) == 1: return output_tuple[0] # Just last_hidden_state
        return output_tuple