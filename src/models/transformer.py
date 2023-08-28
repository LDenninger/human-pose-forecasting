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
from .utils import PointWiseLinear

#####===== Transformer Block ====#####

class SpatioTemporalTransformer(nn.Module):
    """
        Module representing a single Spatial-Temporal Attention Block following: https://arxiv.org/pdf/2004.08692.pdf
    """
    def __init__(
                    self,
                     emb_dim: int,
                     ff_dim: int,
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
        self.ff_dim = ff_dim
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
        self.pointWiseFF = self._build_point_wise_ff()
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
        import ipdb; ipdb.set_trace()
        # Compute spatial and temporal attention separately and update input
        spatialAttentionOut = self.spatialAttention(x) # shape: [batch_size, num_joints, seq_len, emb_dim]
        import ipdb; ipdb.set_trace()
        if self.spatialDropout is not None:
            spatialAttentionOut = self.spatialDropout(spatialAttentionOut)
        temporalAttentionOut = self.temporalAttention(x)
        import ipdb; ipdb.set_trace()
        if self.temporalDropout is not None:
            temporalAttentionOut = self.temporalDropout(temporalAttentionOut)
        # Add spatial and temporal attention
        attnOut = self.layerNorm(x + spatialAttentionOut) + self.layerNorm(x+temporalAttentionOut) # shape: [batch_size, num_joints, seq_len, emb_dim]
        import ipdb; ipdb.set_trace()
        # Point-wise feed-forward layer (point-wise w.r.t. the joints)
        # TODO: Implement this more efficiently by defining projection by hand
        ffOut = self.pointWiseFF(attnOut)
        import ipdb; ipdb.set_trace()
        if self.ffDropout is not None:
            ffOut = self.ffDropout(ffOut)
        ffOut = self.layerNorm(ffOut + attnOut)
        import ipdb; ipdb.set_trace()

        if self.full_return:
            return attnOut, temporalAttentionOut, spatialAttentionOut
        
        return attnOut
        
    def _build_point_wise_ff(self):
        return nn.Sequential(
            nn.Linear(self.emb_dim, self.ff_dim),
            nn.ReLU(),
            nn.Linear(self.ff_dim, self.emb_dim),
        )
    
    def extra_repr(self) -> str:
        return f'emb_size={self.emb_dim}, num_emb={self.num_emb}, temporal_heads={self.temporal_heads}, spatial_heads={self.spatial_heads}, temporal_dropout={self.temporal_dropout}, spatial_dropout={self.spatial_dropout}, ff_dropout={self.ff_dropout}'
    
