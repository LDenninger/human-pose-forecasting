"""
    Transformer blocks used in the pose predictor.
    The used attention mechanisms are implemented in the file attention.py.

    Author: Luis Denninger <l_denninger@uni-bonn.de>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Optional

from .attention import TemporalAttention, SpatialAttention

#####===== Transformer Block ====#####

class SpatioTemporalTransformer(nn.Module):
    """
        Module representing a single Spatial-Temporal Attention Block following: https://arxiv.org/pdf/2004.08692.pdf
    """
    def __init__(
                    self,
                     emb_dim: int,
                     num_emb: int,
                     seq_len: int,
                       temporal_heads: int,
                       spatial_heads: int,
                         temporal_dropout: float = 0.0,
                         spatial_dropout: float =0.0,
                         ff_dropout: float = 0.0,
                          full_return: bool = False

                          ):
        super(SpatioTemporalTransformer, self).__init__()
        # Meta Parameters
        self.emb_dim = emb_dim
        self.num_emb = num_emb
        self.seq_len = seq_len
        self.temporal_heads = temporal_heads
        self.spatial_heads = spatial_heads
        self.temporal_dropout = temporal_dropout
        self.spatial_dropout = spatial_dropout
        self.ff_dropout = ff_dropout

        self.full_return = full_return
        # Define used modules
        self.temporalAttention = TemporalAttention(
            num_emb=num_emb,
            num_tokens=seq_len,
            token_dim=emb_dim,
            num_heads = temporal_heads
        )
        self.spatialAttention = SpatialAttention(
            num_emb=num_emb,
            num_tokens=num_emb,
            token_dim=emb_dim,
            num_heads = spatial_heads
        )
        self.layerNorm = nn.LayerNorm(emb_dim, eps=1e-6)
        self.pointWiseFF = [self._build_point_wise_ff() for _ in range(num_emb)]
        self.temporalDropout = None
        self.spatialDropout = None
        self.ffDropout = None

        if temporal_dropout !=0:
            self.temporalDropout = nn.Dropout(p=temporal_dropout)
        if spatial_dropout!=0:
            self.spatialDropout = nn.Dropout(p=spatial_dropout)
        if ff_dropout!=0:
            self.ffDropout = nn.Dropout(p=ff_dropout)

        

    
    def forward(self, x):
        """
            Forward function for the Spatial-Temporal Attention Block.

            Inputs:
                x: input tensor, shape: [batch_size, seq_len, num_joints, emb_dim]
        """

        # Compute spatial and temporal attention separately and update input
        spatialAttentionOut = self.spatialAttention(x) # shape: [batch_size, num_joints, seq_len, emb_dim]
        if self.spatialDropout is not None:
            spatialAttentionOut = self.spatialDropout(spatialAttentionOut)
        temporalAttentionOut = self.temporalAttention(x)
        if self.temporalDropout is not None:
            temporalAttentionOut = self.temporalDropout(temporalAttentionOut)
        # Add spatial and temporal attention
        attnOut = self.layerNorm(x + spatialAttentionOut) + self.layerNorm(x+temporalAttentionOut) # shape: [batch_size, num_joints, seq_len, emb_dim]
        # Point-wise feed-forward layer (point-wise w.r.t. the joints)
        # TODO: Implement this more efficiently by defining projection by hand

        for i in range(self.num_emb):
            ffOut = self.pointWiseFF[i](attnOut[:,:,i])
            if self.ffDropout is not None:
                ffOut = self.ffDropout(ffOut)
            attnOut[:,:,i] = self.layerNorm(attnOut[:,:,i] + ffOut)
        if self.full_return:
            return attnOut, temporalAttentionOut, spatialAttentionOut
        
        return attnOut
        
    def _build_point_wise_ff(self):
        return nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )
    
    def extra_repr(self) -> str:
        return f'emb_size={self.emb_dim}, num_emb={self.num_emb}, temporal_heads={self.temporal_heads}, spatial_heads={self.spatial_heads}, temporal_dropout={self.temporal_dropout}, spatial_dropout={self.spatial_dropout}, ff_dropout={self.ff_dropout}'
    
#####===== Unused Modules =====#####

class MaskedLinear(nn.Module):
    """
        Copied source code from PyTorch to implement a masked linear layer.
        Specific weights according to the mask are set to zero to mask out inputs.
        This solution might be rather ugly.
    
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, mask=None, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.mask = mask
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        if self.mask is not None:
            self.weight[self.mask] = 0.0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'