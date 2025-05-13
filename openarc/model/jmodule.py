import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Optional

class JCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, config): # Pass full config for flexibility
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size # Dimension of h_t and c_t
        self.config = config

        self.linear_fh = nn.Linear(hidden_size, hidden_size, bias=True) # Operates on h_{t-1}
        self.linear_fx = nn.Linear(input_size, hidden_size, bias=False) # Operates on x_t (bias in linear_fh)

        # Input Gate components (controls what new info to store in c_t)
        self.linear_ih = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear_ix = nn.Linear(input_size, hidden_size, bias=False)

        # Candidate Memory components (c_tilde_t - new candidate values for memory)
        # This is where you can get creative with Conv1D, Pooling, etc.
        # For a start, let's use linear layers, but this can be expanded.
        self.linear_ch = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear_cx = nn.Linear(input_size, hidden_size, bias=False)

        # Output Gate components (controls what to output from c_t to h_t)
        self.linear_oh = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear_ox = nn.Linear(input_size, hidden_size, bias=False)

        self.candidate_activation = getattr(F, str(config.jcell_candidate_activation), torch.tanh)
        self.hidden_activation = getattr(F, str(config.jcell_hidden_activation), torch.tanh)


    def forward(self, x_t: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor]):
        """
        x_t: Current input at timestep t. Shape: (batch_size, input_size)
        states: Tuple (h_prev, c_prev)
            h_prev: Previous hidden state. Shape: (batch_size, hidden_size)
            c_prev: Previous cell state (memory). Shape: (batch_size, hidden_size)
        Returns:
            h_next: Next hidden state. Shape: (batch_size, hidden_size)
            c_next: Next cell state (memory). Shape: (batch_size, hidden_size)
        """
        h_prev, c_prev = states

        # Forget Gate
        f_t = torch.sigmoid(self.linear_fh(h_prev) + self.linear_fx(x_t))

        # Input Gate
        i_t = torch.sigmoid(self.linear_ih(h_prev) + self.linear_ix(x_t))

        # Candidate Memory (c_tilde_t)
        # For now, a simple linear combination:
        c_tilde_t = self.candidate_activation(self.linear_ch(h_prev) + self.linear_cx(x_t))

        # Cell State Update (Memory Update)
        c_next = (f_t * c_prev) + (i_t * c_tilde_t)

        # Output Gate
        o_t = torch.sigmoid(self.linear_oh(h_prev) + self.linear_ox(x_t))

        # Hidden State Update
        h_next = o_t * self.hidden_activation(c_next)

        return h_next, c_next

class JModule(nn.Module):
    def __init__(self, config, num_layers): # Config should contain input_size, hidden_size, etc.
        super().__init__()
        self.config = config
        self.input_size = config.jmodule_input_size
        self.hidden_size = config.jmodule_hidden_size # Dimension of h and c states
        
        # Potentially multiple layers of JCell stacked
        self.num_layers = num_layers
        
        self.cells = nn.ModuleList()
        for i in range(self.num_layers):
            layer_input_size = self.input_size if i == 0 else self.hidden_size
            self.cells.append(JCell(layer_input_size, self.hidden_size, config))
            
        # Dropout between JModule layers if num_layers > 1
        self.dropout = nn.Dropout(getattr(config, 'jmodule_dropout', 0.1)) if self.num_layers > 1 else None

        # Optional: Output projection if the final hidden_size needs to be different
        # or if you want a specific output per timestep (like yt_pred in your NumPy LSTM)
        self.output_projection_dim = getattr(config, 'jmodule_output_projection_dim', -1)
        if self.output_projection_dim > 0:
            self.output_fc = nn.Linear(self.hidden_size, self.output_projection_dim)
        else:
            self.output_fc = nn.Identity() # Output hidden states directly


    def forward(self, x: torch.Tensor, initial_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None):
        """
        x: Input sequence. Shape: (batch_size, seq_len, input_size)
        initial_states: Optional list of (h_0, c_0) tuples, one for each layer.
                        Each h_0, c_0 shape: (batch_size, hidden_size)
        Returns:
            outputs: Sequence of outputs from the last JCell layer.
                     Shape: (batch_size, seq_len, hidden_size or output_projection_dim)
            last_states: List of (h_n, c_n) tuples from all JCell layers.
        """
        batch_size, seq_len, _ = x.shape

        # Initialize states if not provided
        if initial_states is None:
            initial_states = []
            for _ in range(self.num_layers):
                h_0 = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
                c_0 = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
                initial_states.append((h_0, c_0))
        elif len(initial_states) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} initial states, got {len(initial_states)}")

        current_layer_input = x  # Input to the first layer
        last_states_list = []

        for layer_idx in range(self.num_layers):
            cell = self.cells[layer_idx]
            h_prev, c_prev = initial_states[layer_idx]

            # Reshape input for JCell to process the entire sequence at once
            x_reshaped = current_layer_input.reshape(batch_size * seq_len, -1)  

            # Apply JCell to the reshaped input
            h_next, c_next = cell(x_reshaped, (h_prev.repeat_interleave(seq_len, 0), 
                                              c_prev.repeat_interleave(seq_len, 0)))  

            # Reshape the output back to the original sequence format
            current_layer_input = h_next.reshape(batch_size, seq_len, -1)  
            
            last_states_list.append((h_next[-batch_size:], c_next[-batch_size:]))  # Store final states of this layer

            if self.dropout is not None and layer_idx < self.num_layers - 1:  # Apply dropout between layers
                current_layer_input = self.dropout(current_layer_input)

        # The output of the JModule is the sequence of hidden states from the last layer
        final_sequence_output = current_layer_input

        # Apply final output projection
        projected_output = self.output_fc(final_sequence_output)

        return projected_output, last_states_list
    
class _JModule(nn.Module):
    def __init__(self, config, num_layers): 
        super().__init__()
        self.config = config
        self.input_size = config.jmodule_input_size
        self.hidden_size = config.jmodule_hidden_size 
        self.num_layers = num_layers 
        jmodule_dropout = getattr(config, 'jmodule_dropout', 0.1) 

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True, 
            dropout=jmodule_dropout if self.num_layers > 1 else 0, 
            bidirectional=False 
        )
 
        self.output_projection_dim = getattr(config, 'jmodule_output_projection_dim', -1)
        
        if self.output_projection_dim > 0 and self.output_projection_dim != self.hidden_size:
            self.output_fc = nn.Linear(self.hidden_size, self.output_projection_dim)
        elif self.output_projection_dim > 0 and self.output_projection_dim == self.hidden_size:
             self.output_fc = nn.Identity() # No projection needed if dim matches hidden_size
        elif self.output_projection_dim == -1:
             self.output_fc = nn.Identity() # Output LSTM hidden states directly
        else: # Handles self.output_projection_dim = 0 or other invalid negative numbers
             raise ValueError(f"Invalid jmodule_output_projection_dim: {self.output_projection_dim}")


    def forward(self, x: torch.Tensor, initial_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        x: Input sequence. Shape: (batch_size, seq_len, input_size)
        initial_states: Optional tuple (h_0, c_0).
                        h_0 shape: (num_layers, batch_size, hidden_size)
                        c_0 shape: (num_layers, batch_size, hidden_size)
        Returns:
            projected_output: Sequence of outputs after optional projection.
                              Shape: (batch_size, seq_len, hidden_size or output_projection_dim)
            last_states: Tuple (h_n, c_n) from the LSTM layer.
                         h_n shape: (num_layers, batch_size, hidden_size)
                         c_n shape: (num_layers, batch_size, hidden_size)
        """
        batch_size, seq_len, _ = x.shape


        lstm_output, last_states = self.lstm(x, initial_states)

        projected_output = self.output_fc(lstm_output)

        return projected_output, last_states