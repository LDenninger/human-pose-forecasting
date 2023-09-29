"""
    Pose Predictor module which capsules the attention layers.

    Author: Luis Denninger <l_denninger@uni-bonn.de>
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Any, Optional

from .transformer import SpatioTemporalTransformer, getTransformerBlock
from .positional_encoding import PositionalEncodingSinusoidal
from .utils import PointWiseLinear


def getModel(config: Dict[str, Any], device: str = 'cpu') -> nn.Module:
    """
        Construct and return a model given the run configuration.
    """
    if config['model']['type'] == 'st_transformer':
        model = PosePredictor(
            positional_encoding_config=config['model']['positional_encoding'],
            transformer_config=config['model']['transformer'],
            num_joints=config['skeleton']['num_joints'],
            seq_len=config['seq_length'],
            num_blocks=config['model']['num_blocks'],
            emb_dim=config['model']['embedding_dim'],
            joint_dim=config['joint_representation']['joint_dim'],
            input_dropout=config['model']['input_dropout'],
            variable_window=config['variable_window'] if 'variable_window' in config.keys() else False,
            device=device
        ).to(device)
        return model
    else:
        raise NotImplementedError(f'Model type not implemented: {config["model"]["type"]}')

#####====== Pose Predictor =====#####

class PosePredictor(nn.Module):
    """
        Predictor module which encapsule the complete model and attention layers.
    
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
                             incl_abs_position: Optional[bool] = False,
                              variable_window: Optional[bool] = False,
                                device: str = 'cpu'

                    ) -> None:
        """
            Initialize the pose predictor.

            Arguments:
                positional_encoding_config (dict): Configuration for the positional encoding applied to the input.\
                transformer_config (dict): Configuration for the attention layers.
                num_joints (int): Number of joints.
                seq_len (int): Sequence length.
                num_blocks (int): Number of attention blocks.
                emb_dim (int): Embedding dimension.
                joint_dim (int): Dimension of the joint representation.
                input_dropout (float): Dropout rate applied to the input.
                incl_abs_position (bool): Whether to include the absolute position in the input.
        """
        
        super(PosePredictor, self).__init__()
        # Build the model 
        # Initial linear layer for embedding each joint into the embedding space
        self.variable_window = variable_window
        self.device = device
        self.joint_encoder = PointWiseLinear(joint_dim, emb_dim, num_joints)
        if incl_abs_position: 
            self.position_encoder = nn.Linear(3, emb_dim)
            self.position_decoder = nn.Linear(emb_dim, 3)

        #self.jointEncoding = [nn.Linear(joint_dim, emb_dim) for _ in range(num_joints)]
        # Positional encoding on the joints
        self.positionalEncoding = self._resolve_positional_encoding(positional_encoding_config, emb_dim, seq_len)
        self.inputDropout = nn.Dropout(p=input_dropout)
        # Attention blocks
        self.attnBlocks = [getTransformerBlock(transformer_config, num_joints, emb_dim, seq_len) for _ in range(num_blocks)]
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
        self.incl_abs_position = incl_abs_position
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward function of the pose predictor.
            The input is assumed to be given in the following format: [batch, seq_len, joint_dim, joint_repr]
             --> For the attention layers the input requires the format: [batch, joint_dim, seq_len, joint_repr]

            TODO: Make dimensions consistent across all modules to decrease amount of permutations etc.
        """
        # Joint embedding
        out = torch.clone(x)
        if self.incl_abs_position:
            pos = out[:,:,0,:3]
            rot = out[:,:,1:,:]
            rot = self.joint_encoder(rot)
            pos = self.position_encoder(pos)
            out = torch.cat([pos, rot], dim=2)
        else:
            out = self.joint_encoder(out)

        # Temporal positional encoding
        out = self.positionalEncoding(out)
        out = self.inputDropout(out)
        # Attention layers
        out = self.attnBlocks(out)
        # Final decoding to retrieve the original joint representation
        if self.incl_abs_position:
            pos = out[:,:,0,:3]
            rot = out[:,:,1:,:]
            rot = self.joint_decoder(rot)
            pos = self.position_decoder(pos)
            out = torch.cat([pos, rot], dim=2)
        else:
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

    def _resolve_positional_encoding(self, config: dict, emb_dim: int, seq_len: int) -> nn.Module:
        assert config["type"] in ['sin', 'learned'], 'Please provide a valid positional encoding type [sin, learned].'
        if config["type"] =='sin':
            return PositionalEncodingSinusoidal(
                        dim_hidden = emb_dim,
                        n_position = 1000 if self.variable_window else seq_len,
                        device=self.device
            )
        else:
            raise NotImplementedError(f'Positional encoding type not implemented: {type}')
