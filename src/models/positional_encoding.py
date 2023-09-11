"""
    Modules to apply positional encoding to the inputs.

    Author: Luis Denninger <l_denninger@uni-bonn.de>
"""
import torch
import torch.nn as nn
import numpy as np

# Maximal: 50fps
# 100
# 200 

class PositionalEncodingSinusoidal(nn.Module):
    """
        Positional encoding according to "Attention Is All You Need" (https://arxiv.org/pdf/1706.03762.pdf)
    
    """
    def __init__(self, dim_hidden, n_position):
        super(PositionalEncodingSinusoidal, self).__init__()
        # Pre-compute sin and cos tables for positional encoding
        # Generate wave lengths for encoding
        sin_table = np.array([[position / np.power(10000, 2 * (hid_j // 2) / dim_hidden) for hid_j in range(dim_hidden)] for position in range(n_position)])
        # Sinusoidal encoding
        sin_table[:,0::2] = np.sin(sin_table[:,0::2]) # even -> sinus encoding
        sin_table[:,1::2] = np.cos(sin_table[:,1::2]) # odd -> cosinus encoding
        
        self.register_buffer('positional_encoding', torch.FloatTensor(sin_table).unsqueeze(0))

    def forward(self, x):
        """
            Computes the positional encoding.

            Arguments:
                x: Input tensor, shape: [batch_size, seq_len, num_emb, emb_dim]
        """
        return x + self.positional_encoding[:, :x.shape[1]].unsqueeze(-2).clone().detach()
    

class PositionalEncodingStrided(nn.Module):
    """
        A positional encoding that takes different strides in the input and encodes the data accordingly
    """
    def __init__(self, dim_hidden, n_position, max_stride):
        super(PositionalEncodingStrided, self).__init__()
        # Pre-compute sin and cos tables for positional encoding
        # Generate wave lengths for encoding
        sin_table = np.array([[position / np.power(10000, 2 * (hid_j // 2) / dim_hidden) for hid_j in range(dim_hidden)] for position in range(n_position*max_stride)])
        # Sinusoidal encoding
        sin_table[:,0::2] = np.sin(sin_table[:,0::2]) # even -> sinus encoding
        sin_table[:,1::2] = np.cos(sin_table[:,1::2]) # odd -> cosinus encoding
        
        self.register_buffer('positional_encoding', torch.FloatTensor(sin_table).unsqueeze(0))

    def forward(self, x: torch.Tensor, stride: int):
        """
            Computes the positional encoding.

            Arguments:
                x: Input tensor, shape: [batch_size, seq_len, num_emb, emb_dim]
        """
        strided_penc = self.positional_encoding[:, ::stride]
        return x + strided_penc[:, :x.shape[1]].unsqueeze(-2).clone().detach()
