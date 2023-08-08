import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

#####===== High-level Processing Functions =====#####
# These functions are used to perform nested matrix multiplications between two tensors.
# Most of these are used to multiply input tensors with additional dimensions with fixed weight matrices.
# MM: Matrix multiplication, BMM: Batch matrix multiplication, MMM: Multi matrix multiplication, indicating a more elaborate computation scheme than batch-wise computation.
# TODO: Time these function on CPU/GPU. The backend implementation of the Einstein sum can be tricky dependent on how and in which order the batch dimensions are processed.
#  ---> https://davideliu.com/2022/02/20/speed-benchmark-einsum-vs-matmul-in-xl-attention/
# TODO: How it is done right now, the permutation probably causes an overhead...

def multiHeadTemporalMMM(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """ Multi-Head Multi-Matrix Multiplication
        input: mat1 [batch_size, num_emb,  seq_len, emb_dim], mat2 [num_heads, num_emb, emb_dim, proj_dim]
        output: [batch_size, num_heads, num_emb,  seq_len, proj_dim]
    """
    return torch.einsum('bcik,hckj->bchij', mat1, mat2)

def multiHeadSpatialMMVM(mat1: torch.Tensor, vec1: torch.Tensor) -> torch.Tensor:
    """ Multi-Head Multi-Matrix-Vector Multiplication
        input: vec1 [batch_size, num_emb seq_len, emb_dim], mat1 [num_heads, num_emb, proj_dim, emb_dim] (already transposed)
        output: [batch_size, seq_len, num_heads, num_emb, proj_dim]
    
    """
    return torch.einsum('bcik,hcsk->bshci', mat1, vec1)

def multiHeadSpatialMMM(self, mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """ Multi-Head Multi-Matrix Multiplication
        input: mat1 [batch_size, num_emb, seq_len, emb_dim], mat2 [num_heads, emb_dim, proj_dim]
        output: [batch_size, seq_len, num_heads, num_emb , proj_dim]
    """
    return torch.einsum('bisk,hkj->bchij', mat1, mat2)

def multiWeightMMM(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """ 
        input: mat1 [batch_size, num_emb,  seq_len, emb_dim], mat2 [num_emb, emb_dim, proj_dim]
        output: [batch_size, seq_len num_emb, proj_dim]
    """
    return torch.einsum('bcik,ckj->bcij', mat1, mat2)

#####===== Attention Layers ====#####
# Right now the attention layers assume an input of dimension [batch_size, num_joints, seq_len, emb_dim].
# The input might need to be permute to perform more efficient operations. The input dimensions might change later to increase efficiency.

class TemporalAttention(nn.Module):
    """
        Temporal attention layer as described in https://arxiv.org/pdf/2004.08692.pdf.
        TODO:
            1. Implement attention masking as it is required by the method.
            2. Redo dimensions in computation
    
    """
    def __init__(self,
                    emb_dim:int,
                     num_emb: int,
                      num_heads: int,
    ) -> None:
        super().__init__()

        proj_dim = torch.floor(num_emb / num_heads).int()

        self.register_buffer('num_emb', torch.Tensor(num_emb))
        self.register_buffer('emb_dim', torch.Tensor(emb_dim))
        self.register_buffer('num_heads', torch.Tensor(num_heads))
        self.register_buffer('proj_dim', torch.Tensor(proj_dim))

        self.register_parameter('w_query', nn.Parameter(torch.Tensor(num_heads, num_emb, emb_dim, proj_dim)))
        self.register_parameter('W_key', nn.Parameter(torch.Tensor(num_heads, num_emb, emb_dim, proj_dim)))
        self.register_parameter('W_value', nn.Parameter(torch.Tensor(num_heads, num_emb, emb_dim, proj_dim)))
        self.register_parameter('W_output', nn.Parameter(torch.Tensor(num_emb, emb_dim, emb_dim)))

        self._reset_parameters()

    def forward(self, x, mask=None):
        """
            x: input tensor, shape: [batch_size, num_joints, seq_len,  emb_dim]
            TODO: Implement mask functionality
        """
        
        # Compute query, key and value using a the weight matrices.
        query = multiHeadTemporalMMM(x, self.W_query) # shape: [batch,  num_joints, num_heads seq_len, proj_dim]
        key = multiHeadTemporalMMM(x, self.W_key)
        value = multiHeadTemporalMMM(x, self.W_value)

        # Output from scaled dot-product attention: shape: [batch, head, seq_len, proj_dim]
        #  --> permute the output to be [batch, seq_len, head, proj_dim]
        #  --> Flatten head and proj dimensions to retrieve emb_dim
        #  --> Compute temporal attention independently for each joint
        attn_out = torch.stack([torch.flatten(F.scaled_dot_product_attention(query[:,i], key[:,i], value[:,i], mask=mask, dropout=self.dropout).permute(0,2,1,3), start_dim=-2, end_dim=-1) for i in range(self.num_emb)], dim=1)
        # attn_out shape: [batch, num_joints, seq_len, emb_dim]
        # Compute linear embedding of the concatenated heads
        attn_out = multiWeightMMM(attn_out, self.W_output)

        return attn_out

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.W_query)
        self.W_query.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.W_key)
        self.W_key.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.W_value)
        self.W_value.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.W_output)
        self.W_output.bias.data.fill_(0)

#    def expr_repr(self) -> str:
#        return f'num_emb={self.num_emb}, emb_dim={self.emb_dim}, num_heads={self.num_heads}, proj_dim={self.proj_dim}'


class SpatialAttention(nn.Module):
    def __init__(self,
                    emb_dim:int,
                     num_emb: int,
                      num_heads: int,
    ) -> None:
        super(SpatialAttention, self).__init__()

        proj_dim = torch.floor(num_emb / num_heads).int()
        self.register_buffer('num_emb', torch.Tensor(num_emb))
        self.register_buffer('emb_dim', torch.Tensor(emb_dim))
        self.register_buffer('num_heads', torch.Tensor(num_heads))
        self.register_buffer('proj_dim', torch.Tensor(proj_dim))

        # Instead of transposing the query weights everytime, we initialize it in a transposed manner.
        self.register_parameter('w_query', nn.Parameter(torch.Tensor(num_heads, num_emb, proj_dim, emb_dim)))
        self.register_parameter('W_key', nn.Parameter(torch.Tensor(num_heads, emb_dim, proj_dim)))
        self.register_parameter('W_value', nn.Parameter(torch.Tensor(num_heads, emb_dim, proj_dim)))
        self.register_parameter('W_output', nn.Parameter(torch.Tensor(num_emb, emb_dim, emb_dim)))

    def forward(self, x, mask=None):
        """
            x: input tensor, shape: [batch_size, num_joints, seq_len,  emb_dim]
             --> Output should be of same shape
            TODO: Implement mask functionality


        """
        seq_len = x.shape[2]
        
        # Compute query, key and value using a the weight matrices.
        query = multiHeadSpatialMMVM(x, self.W_query) # shape: [batch, seq_len, num_heads, num_joints, proj_dim]
        key = multiHeadSpatialMMM(x, self.W_key)
        value = multiHeadSpatialMMM(x, self.W_value)

        # Output from scaled dot-product attention: shape: [batch, head, num_joints, proj_dim]
        #  --> permute the output to be [batch, num_joints, head, proj_dim]
        #  --> Flatten head and proj dimensions to retrieve emb_dim
        #  --> Compute spatial attention over all joints within a single timestep
        attn_out = torch.stack([torch.flatten(F.scaled_dot_product_attention(query[:,i], key[:,i], value[:,i], mask=mask, dropout=self.dropout).permute(0,2,1,3), start_dim=-2, end_dim=-1) for i in range(seq_len)], dim=2)
        # attn_out shape: [batch, num_joints, seq_len,  emb_dim]
        attn_out = multiWeightMMM(attn_out, self.W_output)

        return attn_out
    
    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.W_query)
        self.W_query.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.W_key)
        self.W_key.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.W_value)
        self.W_value.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.W_output)
        self.W_output.bias.data.fill_(0)

#    def expr_repr(self) -> str:
#        return f'num_emb={self.num_emb}, emb_dim={self.emb_dim}, num_heads={self.num_heads}, proj_dim={self.proj_dim}'


class SpatioTemporalAttentionBlock(nn.Module):
    """
        Module representing a single Spatial-Temporal Attention Block following: https://arxiv.org/pdf/2004.08692.pdf
    """
    def __init__(
                    self,
                     emb_dim: int,
                     num_emb: int,
                       temporal_heads: int,
                       spatial_heads: int,
                         temporal_dropout: float = 0.0,
                         spatial_dropout: float =0.0,
                         ff_dropout: float = 0.0,
                          full_return: bool = False

                          ):
        super(SpatioTemporalAttentionBlock, self).__init__()
        # Register (non-trainable) parameters
        self.register_buffer('emb_dim', torch.Tensor(emb_dim))
        self.register_buffer('num_emb', torch.Tensor(num_emb))
        self.register_buffer('temporal_heads', torch.Tensor(temporal_heads))
        self.register_buffer('spatial_heads', torch.Tensor(spatial_heads))
        self.register_buffer('temporal_dropout', torch.Tensor(temporal_dropout))
        self.register_buffer('spatial_dropout', torch.Tensor(spatial_dropout))
        self.register_buffer('ff_dropout', torch.Tensor(ff_dropout))
        self.full_return = full_return
        # Define used modules
        self.temporalAttention = TemporalAttention(emb_dim, num_emb, temporal_heads)
        self.spatialAttention = SpatialAttention(emb_dim, num_emb, spatial_heads)
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

        

    
    def forward(self, x, mask_temp=None, mask_spat=None):
        """
            Forward function for the Spatial-Temporal Attention Block.

            Inputs:
                x: input tensor, shape: [batch_size, num_joints, seq_len, emb_dim]
        """
        # Compute spatial and temporal attention separately and update input
        spatialAttentionOut = self.spatialAttention(x, mask=mask_spat) # shape: [batch_size, num_joints, seq_len, emb_dim]
        if self.spatialDropout is not None:
            spatialAttentionOut = self.spatialDropout(spatialAttentionOut)
        temporalAttentionOut = self.temporalAttention(x, mask=mask_temp)
        if self.temporalDropout is not None:
            temporalAttentionOut = self.temporalDropout(temporalAttentionOut)
        # Add spatial and temporal attention
        attnOut = self.layerNorm(x + spatialAttentionOut) + self.layerNorm(x+temporalAttentionOut) # shape: [batch_size, num_joints, seq_len, emb_dim]
        # Point-wise feed-forward layer (point-wise w.r.t. the joints)
        # TODO: Implement this more efficiently by defining projection by hand
        for i in self.num_emb:
            ffOut = self.pointWiseFF[i](attnOut[:,i])
            if self.ffDropout is not None:
                ffOut = self.ffDropout(ffOut)
            attnOut[:,i] = self.layerNorm(attnOut[:,i] + ffOut)
        if self.full_return:
            return attnOut, temporalAttentionOut, spatialAttentionOut
        
        return attnOut
        
    def _build_point_wise_ff(self, emb_dim: int):
        return nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )
    
    def extra_repr(self) -> str:
        return f'emb_size={self.emb_dim}, num_emb={self.num_emb}, temporal_heads={self.temporal_heads}, spatial_heads={self.spatial_heads}, temporal_dropout={self.temporal_dropout}, spatial_dropout={self.spatial_dropout}, ff_dropout={self.ff_dropout}'