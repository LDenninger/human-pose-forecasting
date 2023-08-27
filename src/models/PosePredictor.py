"""
    Pose Predictor module which capsules the attention layers.

    Author: Luis Denninger <l_denninger@uni-bonn.de>
"""

import torch
import torch.nn as nn
import math

from .transformer import SpatioTemporalTransformer
from .positional_encoding import PositionalEncodingSinusoidal
from .utils import PointWiseLinear

#####====== Pose Predictor =====#####

class PosePredictor(nn.Module):
    """
        This is a base for a pose predictor.
    
    """

    def __init__(self,
                    positional_encoding_config: dict,
                     transformer_config: dict, 
                      num_joints: int,
                       seq_len: int,
                        num_blocks: int,
                         emb_dim: int,
                          joint_dim: int,
                           input_dropout: float = 0.0,
                    ) -> None:
        
        super(PosePredictor, self).__init__()
        # Build the model 
        # Initial linear layer for embedding each joint into the embedding space

        self.joint_encoder = PointWiseLinear(joint_dim, emb_dim, num_joints)

        #self.jointEncoding = [nn.Linear(joint_dim, emb_dim) for _ in range(num_joints)]
        # Positional encoding on the joints
        self.positionalEncoding = self._resolve_positional_encoding(positional_encoding_config, emb_dim, seq_len)
        self.inputDropout = nn.Dropout(p=input_dropout)
        # Attention blocks
        self.attnBlocks = [self._resolve_transformer(transformer_config, num_joints, emb_dim, seq_len) for _ in range(num_blocks)]
        self.attnBlocks = nn.Sequential(*self.attnBlocks)
        # Final decoding layer to retrieve the original joint representation
        self.joint_decoder = PointWiseLinear(emb_dim, joint_dim, num_joints)
        #self.jointDecoding = [nn.Linear(emb_dim, joint_dim) for _ in range(num_joints)]
        # Save some general parameters
        self.num_joints = num_joints
        self.seq_len = seq_len
        self.num_blocks = num_blocks
        self.emb_dim = emb_dim
        self.joint_dim = joint_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward function of the pose predictor.
            The input is assumed to be given in the following format: [batch, seq_len, joint_dim, joint_repr]
             --> For the attention layers the input requires the format: [batch, joint_dim, seq_len, joint_repr]

            TODO: Make dimensions consistent across all modules to decrease amount of permutations etc.
        """
        # Joint embedding
        out = torch.clone(x)
        out = self.joint_encoder(out)

        # Temporal positional encoding
        out = self.positionalEncoding(out)
        out = self.inputDropout(out)
        # Attention layers
        out = self.attnBlocks(out)
        # Final decoding to retrieve the original joint representation
        out = self.joint_decoder(out)
        # Final residual connection
        out = out + x
        return out

    def encode_joints(self, x: torch.Tensor) -> torch.Tensor:
        """
            Initial joint embedding.
        """
        x = torch.einsum('njk,bsnk->bsnj', self.W_enc, x)
        x += self.b_enc.unsqueeze(0).unsqueeze(0)
        return x

    def decode_joints(self, x: torch.Tensor) -> torch.Tensor:
        """
            Final joint decoding into the original representation.
        """
        x = torch.einsum('njk,bsnk->bsnj', self.W_dec, x)
        x += self.b_dec.unsqueeze(0).unsqueeze(0)
        return x

    def _resolve_transformer(self, 
                                config: dict,
                                 num_joints: int, 
                                  emb_dim: int,
                                   seq_len: int) -> nn.Module:
        assert config['type'] in ['spl', 'vanilla'], 'Please provide a valid transformer type [spl, vanilla].'
        if config['type'] =='spl':
            return SpatioTemporalTransformer(
                        emb_dim = emb_dim,
                        ff_dim = config['ff_dimension'],
                        num_emb = num_joints,
                        seq_len = seq_len,
                        temporal_heads = config['temporal_heads'],
                        spatial_heads = config['spatial_heads'],
                        temporal_dropout=config['temporal_dropout'],
                        spatial_dropout=config['spatial_dropout'],
                        ff_dropout=config['ff_dropout'],
            )
        else:
            raise NotImplementedError(f'Transformer type not implemented: {type}')
    
    def _resolve_positional_encoding(self, config: dict, emb_dim: int, seq_len: int) -> nn.Module:
        assert config["type"] in ['sin', 'learned'], 'Please provide a valid positional encoding type [sin, learned].'
        if config["type"] =='sin':
            return PositionalEncodingSinusoidal(
                        dim_hidden = emb_dim,
                        n_position = seq_len
            )
        else:
            raise NotImplementedError(f'Positional encoding type not implemented: {type}')
