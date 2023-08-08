import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    """
        Positional encoding according to "Attention Is All You Need" (https://arxiv.org/pdf/1706.03762.pdf)
    
    """
    def __init__(self, dim_hidden, n_position):
        
        # Pre-compute sin and cos tables for positional encoding
        # Generate wave lengths for encoding
        sin_table = np.array([[position / np.power(10000, 2 * (hid_j // 2) / dim_hidden) for hid_j in range(dim_hidden)] for position in range(n_position)])
        # Sinusoidal encoding
        sin_table[:,0::2] = np.sin(sin_table[:,0::2]) # even -> sinus encoding
        sin_table[:,1::2] = np.cos(sin_table[:,1::2]) # odd -> cosinus encoding
        
        self.register_buffer('positional_encoding', torch.FloatTensor(sin_table).unsqueeze(0))

    def forward(self, x):
        return x + self.positional_encoding[:, :x.shape[0]].clone().detach()
    
