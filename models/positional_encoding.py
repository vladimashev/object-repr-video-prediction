import math
import torch.nn as nn
import torch

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional encoding 

    Args:
    -----
    d_model: int
        Dimensionality of the slots/tokens
    max_len: int
        Length of the sequence
    """

    def __init__(self, d_model, max_len):
        """
        Initializing the positional encoding
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # initializing embedding
        self.pe = self._get_pe()
        return

    def _get_pe(self):
        """
        Initializing the temporal positional encoding given the encoding mode
        """
        max_len = self.max_len
        d_model = self.d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(1, max_len, d_model)
        return pe

    def forward(self, x):
        """
        Adding the positional encoding to the input tokens of the transformer
        """
        if x.device != self.pe.device:
            self.pe = self.pe.to(x.device)
        batch_size, num_tokens = x.shape[0], x.shape[1]
        cur_pe = self.pe.repeat(batch_size, 1, 1)[:, :num_tokens]

        y = x + cur_pe
        return y