"""
    Functions and modules to be used within models for easier processing.

    Author: Luis Denninger <l_denninger@uni-bonn.de>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Optional, Dict, Any

from .transformer import SpatioTemporalTransformer, SeqSpatioTemporalTransformer, SeqTemporalSpatialTransformer, VanillaTransformer
from .PosePredictor import PosePredictor


#####===== Helper Functions =====#####

def getTransformerBlock(transformer_config: Dict[str, Any],
                    num_joints: int, 
                     emb_dim: int,
                      seq_len: int) -> torch.nn.Module:
    if transformer_config['type'] in ['parallel', 'seq_st', 'seq_ts']:
        if transformer_config['type'] == 'parallel':
            transformer = SeqSpatioTemporalTransformer
        elif transformer_config['type'] =='seq_st':
            transformer = SeqSpatioTemporalTransformer
        elif transformer_config['type'] =='seq_ts':
            transformer = SeqTemporalSpatialTransformer
        return transformer(
                        emb_dim = emb_dim,
                        ff_dim = transformer_config['ff_dimension'],
                        num_emb = num_joints,
                        seq_len = seq_len,
                        temporal_heads = transformer_config['temporal_heads'],
                        spatial_heads = transformer_config['spatial_heads'],
                        temporal_dropout=transformer_config['temporal_dropout'],
                        spatial_dropout=transformer_config['spatial_dropout'],
                        ff_dropout=transformer_config['ff_dropout'],
        )
    elif transformer_config['type'] =='vanilla':
        return VanillaTransformer(
                        emb_dim = emb_dim,
                        ff_dim = transformer_config['ff_dimension'],
                        num_emb = num_joints,
                        seq_len = seq_len,
                        temporal_heads = transformer_config['temporal_heads'],
                        spatial_heads = transformer_config['spatial_heads'],
                        temporal_dropout=transformer_config['temporal_dropout'],
                        spatial_dropout=transformer_config['spatial_dropout'],
                        ff_dropout=transformer_config['ff_dropout'],
    )
    else:
        raise NotImplementedError(f'Transformer type not implemented: {type}')

def getModel(config: Dict[str, Any], device: str = 'cpu') -> nn.Module:
    """
        Construct and return a model given the run configuration.
    """
    if config['model']['type'] == 'baseline':
        model = PosePredictor(
            positional_encoding_config=config['model']['positional_encoding'],
            transformer_config=config['model']['transformer'],
            num_joints=config['skeleton']['num_joints'],
            seq_len=config['seq_length'],
            num_blocks=config['model']['num_blocks'],
            emb_dim=config['model']['embedding_dim'],
            joint_dim=config['joint_representation']['joint_dim'],
            input_dropout=config['model']['input_dropout']
        ).to(device)
        return model
    else:
        raise NotImplementedError(f'Model type not implemented: {config["model"]["type"]}')


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