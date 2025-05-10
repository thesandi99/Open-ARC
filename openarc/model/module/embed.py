# We have this dataset from train and val split

# Example Dataset
# {'task_id': '5e6bbc0b', 'prompt_tensor': tensor([[16, 13, 10,  ..., 15, 15, 15]]), 'prompt_pad_mask_tensor': tensor([[False, False, False,  ...,  True,  True,  True]]), 
# 'target_seq': tensor([15, 15, 15,  ..., 15, 15, 15]), 'is_trunc': False,}

# Creating the embedder handel and perfectly embdding this kind of data set to model 

import torch
import torch.nn as nn
import numpy as np 
import math

from openarc.config.config import config as C

# The OpenEmbedder transforms input sequences of token IDs into rich vector representations. 
# It does this by first looking up a learned embedding for each token, then adding a unique positional signal (sinusoidal positional encoding) 
# to each token's embedding to indicate its position in the sequence. Finally, it applies dropout for regularization. This prepares the input for 
# further processing by a Transformer model.

# We using decoderonly Transformer model That why we use positional signal (sinusoidal positional encoding)

class OpenEmbedder(nn.Module):
    def __init__(self, config=C, embedding_dropout: float = 0.1): 
        """
        Embedder module that combines token embeddings with sinusoidal positional encodings.

        Args:
            config (Config): Configuration object containing vocab_size, hidden_size,
                             pad_token_id, and max_position_embeddings.
            embedding_dropout (float): Dropout rate for the embeddings.
        """
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.pad_token_id = config.pad
        self.max_seq_len_for_pe = config.max_position_embeddings

        self.token_embedding = nn.Embedding(
            self.vocab_size,
            self.hidden_size,
            padding_idx=self.pad_token_id
        )

        # Sinusoidal Positional Encoding
        pe = torch.zeros(self.max_seq_len_for_pe, self.hidden_size)
        position = torch.arange(0, self.max_seq_len_for_pe, dtype=torch.float).unsqueeze(1)
        
        # div_term for sin uses arange(0, hidden_size, 2) -> ceil(hidden_size/2) elements
        # div_term for cos also uses arange(0, hidden_size, 2) -> ceil(hidden_size/2) elements
        div_term = torch.exp(torch.arange(0, self.hidden_size, 2).float() * (-math.log(10000.0) / self.hidden_size))

        pe[:, 0::2] = torch.sin(position * div_term)

        if self.hidden_size % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            # If hidden_size is odd, the last dimension of div_term is not used by cos.
            # Cosine part will have hidden_size // 2 dimensions.
            pe[:, 1::2] = torch.cos(position * div_term[:, :self.hidden_size//2])

        self.register_buffer('pe', pe.unsqueeze(0)) # Shape: (1, max_seq_len, hidden_size)

        self.dropout = nn.Dropout(embedding_dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the embedder.

        Args:
            token_ids (torch.Tensor): Input token IDs. Shape: (batch_size, sequence_length).

        Returns:
            torch.Tensor: Output embeddings. Shape: (batch_size, sequence_length, hidden_size).
        """
        B, S = token_ids.shape
        if S > self.pe.shape[1]:
            raise ValueError(
                f"Sequence length {S} exceeds max_seq_len_for_pe {self.pe.shape[1]}. "
                "You might need to increase `max_position_embeddings` in the config "
                "or truncate input sequences."
            )

        # Ensure token_ids are within valid range
        if torch.any(token_ids >= self.vocab_size) or torch.any(token_ids < 0):
            invalid_tokens = token_ids[(token_ids >= self.vocab_size) | (token_ids < 0)]
            raise ValueError(
                f"Token IDs out of range [0, {self.vocab_size - 1}]. Found: {invalid_tokens.unique().tolist()}. "
                f"Please check your input data and config.vocab_size (currently {self.vocab_size})."
            )


        pe_to_add = self.pe[:, :S, :] # Shape: (1, S, hidden_size)
        embeddings = self.token_embedding(token_ids) # Shape: (B, S, hidden_size)
        
        # Scale embeddings by sqrt(hidden_size) as done in some Transformer models
        embeddings = embeddings * math.sqrt(self.hidden_size)
        embeddings = embeddings + pe_to_add 

        return self.dropout(embeddings)
    
