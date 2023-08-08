import torch
import torch.nn as nn

from config import TransformerConfigBase

from .SPLTransformer import SpatioTemporalAttentionBlock
from .positional_encoding import PositionalEncodingSinusoidal

#####====== Pose Predictor =====#####

class PosePredictor(nn.Module):
    """
        This is a base for a pose predictor.
        The modules used within the pose predictor are easy exchangable to enables
        easier experimenting. 
    
    """

    def __init__(self,
                    positionalEncodingType: str,
                     transformerType: str,
                      transformerConfig: TransformerConfigBase, 
                       num_joints: int,
                        seq_len: int,
                         num_blocks: int,
                          emb_dim: int,
                           joint_dim: int,
                            input_dropout: float = 0.0,
                    ):
        super(PosePredictor, self).__init__()
        # Build the model 
        # Initial linear layer for embedding each joint into the embedding space
        self.jointEncoding = [nn.Linear(joint_dim, emb_dim) for _ in range(num_joints)]
        # Positional encoding on the joints
        self.positionalEncoding = self._resolve_positional_encoding(positionalEncodingType, emb_dim, seq_len)
        self.inputDropout = nn.Dropout(p=input_dropout)
        # Attention blocks
        self.attnBlocks = [self._resolve_transformer(transformerType, num_joints, emb_dim, transformerConfig) for _ in range(num_blocks)]
        self.attnBlocks = nn.Sequential(*self.attnBlocks)
        # Final decoding layer to retrieve the original joint representation
        self.jointDecoding = [nn.Linear(emb_dim, joint_dim) for _ in range(num_joints)]
        # Save some general parameters
        self.register_buffer('num_joints', num_joints)
        self.register_buffer('seq_len', seq_len)
        self.register_buffer('num_blocks', num_blocks)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward function of the pose predictor.
            The input is assumed to be given in the following format: [batch, seq_len, joint_dim, joint_repr]
             --> For the attention layers the input requires the format: [batch, joint_dim, seq_len, joint_repr]

            TODO: Make dimensions consistent across all modules to decrease amount of permutations etc.
        """
        # Preperation
        bs = x.shape[0]
        out = torch.copy(x)
        # Joint embedding
        out = torch.flatten(out, start_dim=0, end_dim=1) # flatten to create a batch dimension
        for i in range(self.num_joints):
            out[:, i] = self.jointEncoding[i](out[:, i])
        out = out.view(bs, self.seq_len, self.num_joints, -1) # unflatten to previous shape
        # Temporal positional encoding
        out = self.positionalEncoding(torch.flatten(out, start_dim=-2, end_dim=-1))
        out = out.view(bs, self.seq_len, self.num_joints, -1)
        out = self.inputDropout(out)
        # Attention layers
        out = self.attnBlocks(out)
        # Final decoding to retrieve the original joint representation
        out = torch.flatten(out, start_dim=0, end_dim=1) # flatten to create a batch dimension
        for i in range(self.num_joints):
            out[:, i] = self.jointDecoding[i](out[:, i])
        out = out.view(bs, self.seq_len, self.num_joints, -1) # unflatten to previous shape
        # Final residual connection
        out = out + x
        return out


    def _resolve_transformer(self, 
                                type: str, 
                                 num_joints: int, 
                                  emb_dim:int, 
                                   config: TransformerConfigBase) -> nn.Module:
        assert type in ['spl', 'vanilla']
        if type =='spl':
            return SpatioTemporalAttentionBlock(
                        emb_dim = emb_dim,
                        num_emb = num_joints,
                        temporal_heads = config.TEMPORAL_HEADS,
                        spatial_heads = config.SPATIAL_HEADS,
                        temporal_dropout=config.TEMPORAL_DROPOUT,
                        spatial_dropout=config.SPATIAL_DROPOUT,
                        ff_dropout=config.FF_DROPOUT,
            )
        elif type == 'vanilla':
            return nn.Transformer
        else:
            raise NotImplementedError(f'Transformer type not implemented: {type}')
    
    def _resolve_positional_encoding(self, type: str, emb_dim: int, seq_len: int) -> nn.Module:
        assert type in ['sinusoidal', 'learned']
        if type =='sinusoidal':
            return PositionalEncodingSinusoidal(
                        dim_hidden = emb_dim,
                        n_position = seq_len
            )
        else:
            raise NotImplementedError(f'Positional encoding type not implemented: {type}')
        
