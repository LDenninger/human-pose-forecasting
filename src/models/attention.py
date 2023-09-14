"""
    This files contains the used attention mechanisms.

    We assume that the base shape of input vectors is [batch_size, seq_len, num_joints, emb_dim]

    Author: Luis Denninger <l_denninger@uni-bonn.de>

"""
import torch
import torch.nn as nn
import math
from abc import abstractmethod


class AttentionBase(nn.Module):
    def __init__(self,
                  num_tokens: int,
                   token_dim: int,
                    num_heads: int) -> None:

        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = self.token_dim // self.num_heads
        self.register_buffer('mask', None)
        return
    
    @abstractmethod
    def forward(self, x:torch.Tensor, mask:torch.Tensor = None) -> torch.Tensor:
        pass

    def scaled_dot_product_attention(self, 
                                      query:torch.Tensor, 
                                       key: torch.Tensor, 
                                        value: torch.Tensor, 
                                          ) -> torch.Tensor:
        """
            Function to perform a scaled dot product attention on the raw input tensor. 
            Leading dimensions are squeezed into one dimension and later reshaped to the original shape.
            input: 
                query [..., num_tokens, token_dim], 
                key [..., num_tokens, token_dim], 
                value [..., num_tokens, token_dim], 
                mask [num_tokens, num_tokens]
            output: [..., num_tokens, token_dim]
        """
        shape = query.shape
        query, key, value = query.reshape(-1, shape[-2], shape[-1]), key.reshape(-1, shape[-2], shape[-1]), value.reshape(-1, shape[-2], shape[-1])
        attn = torch.bmm(query, torch.transpose(key, 1, 2)) * (self.token_dim ** -0.5) 
        if self.mask is not None:
            attn += self.mask.unsqueeze(0)
        attn = torch.softmax(attn, dim=-1)
        out = torch.bmm(attn, value)
        out = out.view(*shape)
        return out
    
    def multi_head_linear_embedding(self, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
            Computes the linear embedding of a batched input and a weight matrix with multiple heads.
            Leading dimension are squeezed into one dimension and later reshaped to the original shape.

            input:
                x [..., num_tokens, token_dim], W [num_tokens, token_dim]
            output: [..., num_heads, num_tokens, head_dim]
        """
        out = self.linear_embedding(x, W)
        out = out.view(*out.shape[:-1], self.num_heads, self.head_dim)
        out = torch.transpose(out, -3, -2)
        return out
    
    def linear_embedding(self, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
            Computes the linear embedding of a batched input and a weight matrix.
            Leading dimension are squeezed into one dimension and later reshaped to the original shape.

            input:
                x [..., num_tokens, token_dim], W [num_tokens, token_dim]
            output: [..., num_tokens, token_dim]
        """
        shape = x.shape
        x = x.view(-1, shape[-2], shape[-1])
        out = torch.einsum('bij,jk->bik', x, W)
        out = out.view(*shape)
        return out
    
    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.W_query)
        nn.init.xavier_uniform_(self.W_key)
        nn.init.xavier_uniform_(self.W_value)
        nn.init.xavier_uniform_(self.W_output)


class TemporalAttention(AttentionBase):
    def __init__(self,
                 num_emb: int,
                  num_tokens: int,
                   token_dim: int,
                    num_heads: int) -> None:

        super().__init__(num_tokens, token_dim, num_heads)
        self.num_emb = num_emb
        self.register_parameter('W_query', nn.Parameter(torch.Tensor(num_emb, token_dim, token_dim)))
        self.register_parameter('W_key', nn.Parameter(torch.Tensor(num_emb, token_dim, token_dim)))
        self.register_parameter('W_value', nn.Parameter(torch.Tensor(num_emb, token_dim, token_dim)))
        self.register_parameter('W_output', nn.Parameter(torch.Tensor(num_emb, token_dim, token_dim)))
        self.set_mask()
        self._reset_parameters()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Apply temporal attention to the input tensor.

            Input:
                x [batch_size, seq_len, num_joints, emb_dim]
        """
        x = x.permute(0, 2, 1, 3) # [batch_size, num_joints, seq_len, emb_dim]
        Q = self.multi_head_linear_embedding(x, self.W_query) # [batch_size, num_joints, num_tokens, token_dim]
        K = self.multi_head_linear_embedding(x, self.W_key)
        V = self.multi_head_linear_embedding(x, self.W_value)
        
        out = self.scaled_dot_product_attention(Q, K, V)
        out = torch.transpose(out, -3, -2)
        out = torch.flatten(out, start_dim=-2, end_dim=-1)
        out = self.linear_embedding(out, self.W_output) 
        out = out.permute(0, 2, 1, 3) 

        return out
    
    
    def linear_embedding(self, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
            Computes the linear embedding of a batched input and a weight matrix.
            Leading dimension are squeezed into one dimension and later reshaped to the original shape.

            input:
                x [..., num_tokens, token_dim], W [num_emb, num_tokens, token_dim]
            output: [..., num_tokens, token_dim]
        """

        shape = x.shape
        x = x.view(-1, shape[-3], shape[-2], shape[-1])
        out = torch.einsum('bnij,njk->bnik', x, W)
        out = out.view(*shape)
        return out

    def set_mask(self) -> None:
        self.mask = torch.triu(torch.ones(self.num_tokens, self.num_tokens), diagonal=1) * 1e9

class SpatialAttention(AttentionBase):
    def __init__(self,
                 num_emb: int,
                  num_tokens: int,
                   token_dim: int,
                    num_heads: int) -> None:
        super().__init__(num_tokens, token_dim, num_heads)
        self.num_emb = num_emb
        self.register_parameter('W_query', nn.Parameter(torch.Tensor(num_emb, token_dim, token_dim)))
        self.register_parameter('W_key', nn.Parameter(torch.Tensor(token_dim, token_dim)))
        self.register_parameter('W_value', nn.Parameter(torch.Tensor(token_dim, token_dim)))
        self.register_parameter('W_output', nn.Parameter(torch.Tensor(token_dim, token_dim)))
        self._reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Apply spatial attention to the input tensor.

            Input:
                x [batch_size, seq_len, num_joints, emb_dim]
        """
        Q = self.multi_head_query_embedding(x, self.W_query) # [batch_size, seq_len, num_heads, num_joints, head_dim]
        K = self.multi_head_linear_embedding(x, self.W_key) # [batch_size, seq_len, num_heads, num_joints, head_dim]
        V = self.multi_head_linear_embedding(x, self.W_value) # [batch_size, seq_len, num_heads, num_joints, head_dim]

        out = self.scaled_dot_product_attention(Q, K, V)
        out = torch.transpose(out, -3, -2)
        out = torch.flatten(out, start_dim=-2, end_dim=-1)
        out = self.linear_embedding(out, self.W_output)

        return out
    
    def multi_head_query_embedding(self, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
            Computes the linear embedding of a batched input and a weight matrix with multiple heads for the query.
            Leading dimension are squeezed into one dimension and later reshaped to the original shape.

            input:
                x [..., num_tokens, token_dim], W [num_emb, num_tokens, token_dim]
            output: [..., num_emb, num_tokens, token_dim]
        """

        out = self.query_embedding(x, W)    
        out = out.view(*out.shape[:-1], self.num_heads, self.head_dim)
        out = torch.transpose(out, -3, -2)
        return out
    
    def query_embedding(self, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
            Embed the query using different embedding functions for each joint.
        """

        shape = x.shape
        x = x.view(-1,shape[-3], shape[-2], shape[-1])
        out = torch.einsum('nik,bjnk->bjni', torch.transpose(W, -2, -1), x) # shape [batch_size*seq_len, num_joints, token_dim]
        out = out.view(*shape) # shape [batch_size, seq_len, num_joints, token_dim]
        return out

class VanillaAttention(AttentionBase):

    def __init__(self,
                num_tokens: int,
                token_dim: int,
                    num_heads: int) -> None:
        super().__init__(num_tokens, token_dim, num_heads)
        self.register_parameter('W_query', nn.Parameter(torch.Tensor(token_dim, token_dim)))
        self.register_parameter('W_key', nn.Parameter(torch.Tensor(token_dim, token_dim)))
        self.register_parameter('W_value', nn.Parameter(torch.Tensor(token_dim, token_dim)))
        self.register_parameter('W_output', nn.Parameter(torch.Tensor(token_dim, token_dim)))
        self._reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Apply temporal attention to the input tensor.

            Input:
                x [batch_size, seq_len, num_joints * emb_dim]
        """
        Q = self.multi_head_linear_embedding(x, self.W_query) # [batch_size,num_tokens, token_dim]
        K = self.multi_head_linear_embedding(x, self.W_key)
        V = self.multi_head_linear_embedding(x, self.W_value)

        out = self.scaled_dot_product_attention(Q, K, V)
        out = self.linear_embedding(out, self.W_output)
        return out
    
    def set_mask(self) -> None:
        self.mask = torch.triu(torch.ones(self.num_tokens, self.num_tokens), diagonal=1) * 1e9