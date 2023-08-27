"""
    Functions and modules to be used within models for easier processing.

    Author: Luis Denninger <l_denninger@uni-bonn.de>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Optional

#####===== Helper Functions =====#####

def generateTemporalMask(seq_len: int, device: Optional[torch.device]) -> torch.Tensor:
    """ Generate a temporal mask for a given sequence length to prevent leakage from future timesteps.
        input: seq_len (int)
        output: [seq_len, seq_len]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=1)
    if device is not None:
        mask = mask.to(device)
    return mask


#####===== High-level Processing Functions =====#####
"""
    These functions are used to perform nested matrix multiplications between two tensors.
    Most of these are used to multiply input tensors with additional dimensions with fixed weight matrices.

    For further insights of the performed matrix multiplications please refer to our test functions here: /test/model_tests.py
    There each intendented operation is explicitely performed in a nested for loops.

    MM: Matrix multiplication, BMM: Batch matrix multiplication, MMM: Multi matrix multiplication, indicating a more elaborate computation scheme than batch-wise computation.
    
    TODO: Time these function on CPU/GPU. The backend implementation of the Einstein sum can be tricky dependent on how and in which order the batch dimensions are processed.
     ---> https://davideliu.com/2022/02/20/speed-benchmark-einsum-vs-matmul-in-xl-attention/
"""

def multiHeadTemporalMMM(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """ Multi-Head Multi-Matrix Multiplication
        input: mat1 [batch_size, num_emb,  seq_len, emb_dim], mat2 [num_heads, num_emb, emb_dim, proj_dim]
        output: [batch_size, num_heads, num_emb,  seq_len, proj_dim]
    """
    return torch.einsum('bcik,hckj->bhcij', mat1, mat2)

def multiHeadSpatialMMVM(mat1: torch.Tensor, vec1: torch.Tensor) -> torch.Tensor:
    """ Multi-Head Multi-Matrix-Vector Multiplication
        input: vec1 [batch_size, num_emb seq_len, emb_dim], mat1 [num_heads, num_emb, proj_dim, emb_dim] (already transposed)
        output: [batch_size, seq_len, num_heads, num_emb, proj_dim]
    
    """
    return torch.einsum('hcik,bcsk->bshci', mat1, vec1)

def multiHeadSpatialMMM(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """ Multi-Head Multi-Matrix Multiplication
        input: mat1 [batch_size, num_emb, seq_len, emb_dim], mat2 [num_heads, emb_dim, proj_dim]
        output: [batch_size, seq_len, num_heads, num_emb , proj_dim]
    """
    return torch.einsum('bisk,hkj->bshij', mat1, mat2)

def multiWeightMMM(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """ 
        input: mat1 [batch_size, num_emb,  seq_len, emb_dim], mat2 [num_emb, emb_dim, emb_dim]
        output: [batch_size, seq_len num_emb, emb_dim]
    """
    return torch.einsum('bcik,ckj->bicj', mat1, mat2)


#####===== Helper Modules =====#####

class PointWiseLinear(nn.Module):
    """
        Point-wise linear layer.
        This module computes a a separate linear projection for each point in the input.
    """
    def __init__(self,
                  in_features: int,
                   out_features: int,
                     num_points: int,
                       bias: Optional[bool] = True,
                          ) -> None:
        super(PointWiseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_points = num_points
        self.weight = nn.Parameter(torch.Tensor(num_points, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_points, out_features))
        else:
            self.bias = None
        self.reset_parameters()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward function for the point-wise linear layer.

            Arguments:
                x (torch.Tensor): batched input tensor. Additional dimensions are squeezed and reshaped. [batch_size,..., num_points, in_features]
        """
        shape = x.shape
        x = x.view(-1, self.num_points, self.in_features)
        x = torch.einsum('njk,bnk->bnj', self.weight, x)
        if self.bias is not None:
            x = x + self.bias.unsqueeze(0)
        x = x.reshape(*shape[:-1], self.out_features)
        return x

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, num_points={self.num_points} bias={self.bias is not None}'