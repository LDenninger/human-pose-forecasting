"""
    Transformer blocks used in the pose predictor.
    The used attention mechanisms are implemented in the file attention.py.

    Author: Luis Denninger <l_denninger@uni-bonn.de>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Optional, Dict, Any

from .attention import TemporalAttention, SpatialAttention, VanillaAttention
from .utils import PointWiseLinear

def getTransformerBlock(transformer_config: Dict[str, Any],
                        num_joints: int, 
                         emb_dim: int,
                          seq_len: int,
                           return_attention: Optional[bool] = False) -> torch.nn.Module:
    if transformer_config['type'] in ['parallel', 'seq_st', 'seq_ts']:
        if transformer_config['type'] == 'parallel':
            transformer = SpatioTemporalTransformer
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
                        full_return=return_attention
        )
    elif transformer_config['type'] =='vanilla':
        return VanillaTransformer(
                        emb_dim = emb_dim*num_joints,
                        ff_dim = transformer_config['ff_dimension'],
                        num_tokens = seq_len,
                        heads = transformer_config['heads'],
                        attention_dropout=transformer_config['attention_dropout'],
                        ff_dropout=transformer_config['ff_dropout'],
                        full_return=return_attention
    )
    else:
        raise NotImplementedError(f'Transformer type not implemented: {type}')


#####===== Transformer Block ====#####

class VanillaTransformer(nn.Module):
    def __init__(self,
                  emb_dim: int,
                  ff_dim: int,
                  num_tokens: int,
                    heads: int,
                     attention_dropout: float = 0.0,
                     ff_dropout: float = 0.0,
                      full_return: bool = False):
        super(VanillaTransformer, self).__init__()
        # Meta Parameters
        self.emb_dim = emb_dim
        self.ff_dim = ff_dim
        self.ff_dropout = ff_dropout

        self.full_return = full_return
        # Define used modules
        self.vanillaAttention = VanillaAttention(
            num_tokens=num_tokens,
            token_dim=emb_dim,
            num_heads = heads,
        )
        self.layerNorm = nn.LayerNorm(emb_dim, eps=1e-6)
        self.pointWiseFF = self._build_point_wise_ff()

        if attention_dropout !=0:
            self.attentionDropout = nn.Dropout(p=attention_dropout)
        if ff_dropout!=0:
            self.ffDropout = nn.Dropout(p=ff_dropout)

    
    def forward(self, x):
        """
            Forward function for the Spatial-Temporal Attention Block.

            Inputs:
                x: input tensor, shape: [batch_size, seq_len, num_joints, emb_dim]
        """
        if self.full_return and not torch.is_tensor(x):
            x = x[0]
        shape = x.shape
        x = torch.flatten(x, start_dim=-2, end_dim=-1)
        # Compute spatial and temporal attention separately and update input
        attentionOut = self.vanillaAttention(x)
        if self.full_return:
            attn_return = attentionOut.detach().cpu()
        if self.attentionDropout is not None:
            attentionOut = self.attentionDropout(attentionOut)
        # Add spatial and temporal attention
        attnOut = self.layerNorm(x + attentionOut)
        # Point-wise feed-forward layer (point-wise w.r.t. the joints)
        ffOut = self.pointWiseFF(attnOut)
        if self.ffDropout is not None:
            ffOut = self.ffDropout(ffOut)
        ffOut = self.layerNorm(ffOut + attnOut)

        if self.full_return:
            return ffOut, attn_return
        
        ffOut = torch.reshape(ffOut, shape)
        return ffOut
        
    def _build_point_wise_ff(self):
        return nn.Sequential(
            nn.Linear(self.emb_dim, self.ff_dim),
            nn.ReLU(),
            nn.Linear(self.ff_dim, self.emb_dim),
        )
    
    def extra_repr(self) -> str:
        return f'emb_size={self.emb_dim}, num_emb={self.num_emb}, temporal_heads={self.temporal_heads}, spatial_heads={self.spatial_heads}, temporal_dropout={self.temporal_dropout}, spatial_dropout={self.spatial_dropout}, ff_dropout={self.ff_dropout}'
    

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
        if self.full_return and not torch.is_tensor(x):
            x = x[0]
        # Compute spatial and temporal attention separately and update input
        spatialAttentionOut = self.spatialAttention(x) # shape: [batch_size, num_joints, seq_len, emb_dim]
        if self.full_return:
            spatial_attn_return = spatialAttentionOut.detach().cpu()
        if self.spatialDropout is not None:
            spatialAttentionOut = self.spatialDropout(spatialAttentionOut)
        temporalAttentionOut = self.temporalAttention(x)
        if self.full_return:
            temporal_attn_return = temporalAttentionOut.detach().cpu()
        if self.temporalDropout is not None:
            temporalAttentionOut = self.temporalDropout(temporalAttentionOut)
        # Add spatial and temporal attention
        attnOut = self.layerNorm(x + spatialAttentionOut) + self.layerNorm(x+temporalAttentionOut) # shape: [batch_size, num_joints, seq_len, emb_dim]
        # Point-wise feed-forward layer (point-wise w.r.t. the joints)
        # TODO: Implement this more efficiently by defining projection by hand
        ffOut = self.pointWiseFF(attnOut)
        if self.ffDropout is not None:
            ffOut = self.ffDropout(ffOut)
        ffOut = self.layerNorm(ffOut + attnOut)

        if self.full_return:
            return ffOut, temporal_attn_return, spatial_attn_return
        
        return ffOut
        
    def _build_point_wise_ff(self):
        return nn.Sequential(
            nn.Linear(self.emb_dim, self.ff_dim),
            nn.ReLU(),
            nn.Linear(self.ff_dim, self.emb_dim),
        )
    
    def extra_repr(self) -> str:
        return f'emb_size={self.emb_dim}, num_emb={self.num_emb}, temporal_heads={self.temporal_heads}, spatial_heads={self.spatial_heads}, temporal_dropout={self.temporal_dropout}, spatial_dropout={self.spatial_dropout}, ff_dropout={self.ff_dropout}'
    
class SeqSpatioTemporalTransformer(nn.Module):

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
        super(SeqSpatioTemporalTransformer, self).__init__()
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
        self.spatialAttention = SpatialAttention(
            num_emb=num_emb,
            num_tokens=num_emb,
            token_dim=emb_dim,
            num_heads = spatial_heads
        )
        self.spatialPointWiseFF = self._build_point_wise_ff()

        self.temporalAttention = TemporalAttention(
            num_emb=num_emb,
            num_tokens=seq_len,
            token_dim=emb_dim,
            num_heads = temporal_heads
        )
        self.temporalPointWiseFF = self._build_point_wise_ff()


        self.layerNorm = nn.LayerNorm(emb_dim, eps=1e-6)
        self.temporalDropout = None
        self.spatialDropout = None
        self.ffDropout = None

        if temporal_dropout !=0:
            self.temporalDropout = nn.Dropout(p=temporal_dropout)
        if spatial_dropout!=0:
            self.spatialDropout = nn.Dropout(p=spatial_dropout)
        if ff_dropout!=0:
            self.ffDropout = nn.Dropout(p=ff_dropout)

        
        return
    
    def forward(self, x):
        """
            Forward function for the Spatial-Temporal Attention Block.

            Inputs:
                x: input tensor, shape: [batch_size, seq_len, num_joints, emb_dim]
        """
        if self.full_return and not torch.is_tensor(x):
            x = x[0]
        # Compute spatial and temporal attention separately and update input
        spatialAttentionOut = self.spatialAttention(x) # shape: [batch_size, num_joints, seq_len, emb_dim]
        if self.full_return:
            spatial_attn_return = spatialAttentionOut.detach().cpu()
        if self.spatialDropout is not None:
            spatialAttentionOut = self.spatialDropout(spatialAttentionOut)
        spatialOut = self.spatialPointWiseFF(spatialAttentionOut)
        if self.full_return:
            temporal_attn_return = temporalAttentionOut.detach().cpu()
        if self.ffDropout is not None:
            spatialOut = self.ffDropout(spatialOut)
        spatialOut = self.layerNorm(spatialOut + x)

        temporalAttentionOut = self.temporalAttention(spatialOut)
        if self.temporalDropout is not None:
            temporalAttentionOut = self.temporalDropout(temporalAttentionOut)
        temporalOut = self.temporalPointWiseFF(temporalAttentionOut)
        if self.ffDropout is not None:
            temporalOut = self.ffDropout(temporalOut)
        temporalOut = self.layerNorm(temporalOut + spatialOut)
        
        if self.full_return:

            return temporalOut, temporal_attn_return, spatial_attn_return
        
        return temporalOut
    
    def _build_point_wise_ff(self):
        return nn.Sequential(
            nn.Linear(self.emb_dim, self.ff_dim),
            nn.ReLU(),
            nn.Linear(self.ff_dim, self.emb_dim),
        )
    
    def extra_repr(self) -> str:
        return f'emb_size={self.emb_dim}, num_emb={self.num_emb}, temporal_heads={self.temporal_heads}, spatial_heads={self.spatial_heads}, temporal_dropout={self.temporal_dropout}, spatial_dropout={self.spatial_dropout}, ff_dropout={self.ff_dropout}'
    

class SeqTemporalSpatialTransformer(nn.Module):

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
        super(SeqTemporalSpatialTransformer, self).__init__()
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
        self.temporalPointWiseFF = self._build_point_wise_ff()

        self.spatialAttention = SpatialAttention(
            num_emb=num_emb,
            num_tokens=num_emb,
            token_dim=emb_dim,
            num_heads = spatial_heads
        )
        self.spatialPointWiseFF = self._build_point_wise_ff()


        self.layerNorm = nn.LayerNorm(emb_dim, eps=1e-6)
        self.temporalDropout = None
        self.spatialDropout = None
        self.ffDropout = None

        if temporal_dropout !=0:
            self.temporalDropout = nn.Dropout(p=temporal_dropout)
        if spatial_dropout!=0:
            self.spatialDropout = nn.Dropout(p=spatial_dropout)
        if ff_dropout!=0:
            self.ffDropout = nn.Dropout(p=ff_dropout)

        
        return
    
    def forward(self, x):
        """
            Forward function for the Spatial-Temporal Attention Block.

            Inputs:
                x: input tensor, shape: [batch_size, seq_len, num_joints, emb_dim]
        """
        if self.full_return and not torch.is_tensor(x):
            x = x[0]
        # Compute spatial and temporal attention separately and update input
        temporalAttentionOut = self.temporalAttention(x)
        if self.full_return:
            temporal_attn_return = temporalAttentionOut.detach().cpu()
        if self.temporalDropout is not None:
            temporalAttentionOut = self.temporalDropout(temporalAttentionOut)
        temporalOut = self.temporalPointWiseFF(temporalAttentionOut)
        if self.ffDropout is not None:
            temporalOut = self.ffDropout(temporalOut)
        temporalOut = self.layerNorm(temporalOut + x)

        spatialAttentionOut = self.spatialAttention(temporalOut) # shape: [batch_size, num_joints, seq_len, emb_dim]
        if self.full_return:
            spatial_attn_return = spatialAttentionOut.detach().cpu()
        if self.spatialDropout is not None:
            spatialAttentionOut = self.spatialDropout(spatialAttentionOut)
        spatialOut = self.spatialPointWiseFF(spatialAttentionOut)
        if self.ffDropout is not None:
            spatialOut = self.ffDropout(spatialOut)
        spatialOut = self.layerNorm(spatialOut + temporalOut)
        
        if self.full_return:
            return spatialOut, temporal_attn_return, spatial_attn_return
        
        return spatialOut
    
    def _build_point_wise_ff(self):
        return nn.Sequential(
            nn.Linear(self.emb_dim, self.ff_dim),
            nn.ReLU(),
            nn.Linear(self.ff_dim, self.emb_dim),
        )
    
    def extra_repr(self) -> str:
        return f'emb_size={self.emb_dim}, num_emb={self.num_emb}, temporal_heads={self.temporal_heads}, spatial_heads={self.spatial_heads}, temporal_dropout={self.temporal_dropout}, spatial_dropout={self.spatial_dropout}, ff_dropout={self.ff_dropout}'
    