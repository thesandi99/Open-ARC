# model.py

import torch
import torch.nn as nn
from typing import Optional, Tuple, List # Added List

from openarc.model.module.embed import OpenEmbedder
from openarc.model.jmodule import JModule 
from openarc.model.decoder import OpenDecoder 
from openarc.config.config import config as C 


class OpenARC(nn.Module):
    def __init__(self, config=C): # Use the imported global config or pass an instance
        super(OpenARC, self).__init__()
        self.config = config

        self.embedder = OpenEmbedder(config)
        self.decoder = OpenDecoder(config) 

        self.jmodule_num_internal_layers = config.jmodule_recurrent_layers
        self.jmodule = JModule(config, num_layers=self.jmodule_num_internal_layers)

        # Layer normalization before the final LM head
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # LM head for final token predictions
        self.lm_head = nn.Linear(config.hidden_size, config.output_size)


    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        
        # KV cache for the attention mechanisms within OpenDecoder
        past_key_values_attn: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        
        # Initial states for the JModule (recurrent states h_0, c_0)
        initial_jmodule_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: Optional[bool] = None, # Controls KV caching for attention
        output_attentions: Optional[bool] = None,
        output_hidden_states_all: Optional[bool] = None,
        
    ):
        if use_cache is None:
            use_cache = getattr(self.config, "use_cache", False)
        if output_attentions is None:
            output_attentions = getattr(self.config, "output_attentions", False)
        if output_hidden_states_all is None:
            output_hidden_states_all = getattr(self.config, "output_hidden_states_all", False)


        # Embed input
        embedded_input = self.embedder(input_ids)

        # Pass embedded input 
        decoder_outputs_tuple = self.decoder(
            hidden_states=embedded_input,
            attention_mask=attention_mask, 
            past_key_values=past_key_values_attn, 
            use_cache=use_cache,
            output_attentions=output_attentions,
        )


        if isinstance(decoder_outputs_tuple, tuple):
            last_hidden_state_from_decoder = decoder_outputs_tuple[0]
            next_past_key_values_attn = decoder_outputs_tuple[1] if use_cache and len(decoder_outputs_tuple) > 1 else None
        else: # If OpenDecoder returns only hidden_states
            last_hidden_state_from_decoder = decoder_outputs_tuple
            next_past_key_values_attn = None


        # Pass decoder output 
        jmodule_output_sequence, last_jmodule_states_list = self.jmodule(
            x=last_hidden_state_from_decoder,
            initial_states=initial_jmodule_states
        )

        # Apply final layer normalization
        if jmodule_output_sequence.shape[-1] != self.config.hidden_size:
            if self.jmodule.output_projection_dim > 0 and self.jmodule.output_projection_dim != self.config.hidden_size:
                print(f"Warning: JModule output projection dim {self.jmodule.output_projection_dim} "
                       f"differs from final_layer_norm/lm_head input dim {self.config.hidden_size}. "
                       "Ensure JModule's output_fc projects to config.hidden_size if this is not intended.")
            
            elif self.jmodule.output_projection_dim == -1 and self.jmodule.hidden_size != self.config.hidden_size:
                # This case should ideally not happen if jmodule_hidden_size is set based on config.hidden_size
                print(f"Warning: JModule internal hidden size {self.jmodule.hidden_size} "
                       f"differs from final_layer_norm/lm_head input dim {self.config.hidden_size} "
                       "and no projection is active in JModule to match it.")

        # The output from JModule becomes the input for the final normalization and LM head
        sequence_output_for_lm_head = jmodule_output_sequence
        normalized_output = self.final_layer_norm(sequence_output_for_lm_head)
        
        # Apply final fully connected layer (LM head)
        logits = self.lm_head(normalized_output) # (batch_size, seq_len, output_size/vocab_size)

        model_outputs = (logits,)

        if use_cache: # For attention KV cache
            model_outputs += (next_past_key_values_attn,)
        
        # Add JModule's last states. This makes the output signature different from standard HF.
        model_outputs += (last_jmodule_states_list,)

        if len(model_outputs) == 1:
            return model_outputs[0]
        
        return model_outputs