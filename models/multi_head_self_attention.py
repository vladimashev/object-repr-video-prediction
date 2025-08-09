import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """ 
    Self-Attention module

    Args:
    -----
    token_dim: int
        Dimensionality of the tokens in the transformer
    inner_dim: int
        Dimensionality used for attention
    """

    def __init__(self, token_dim, attn_dim, num_heads):
        """ """
        super().__init__()
        self.token_dim = token_dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        assert num_heads >= 1
        assert attn_dim % num_heads == 0, f"{attn_dim = } must be divisible by {num_heads = }..."
        self.head_dim = attn_dim // num_heads

        # query, key and value projections
        self.q = nn.Linear(token_dim, attn_dim, bias=False)
        self.k = nn.Linear(token_dim, attn_dim, bias=False)
        self.v = nn.Linear(token_dim, attn_dim, bias=False)

        # output projection
        self.out_proj = nn.Linear(attn_dim, token_dim, bias=False)
        return
    
    def attention(self, query, key, value):
        """
        Computing self-attention

        All (q,k,v) ~ (B, N, D)
        """
        scale = (query.shape[-1]) ** (-0.5)

        # similarity between each query and the keys
        similarity = torch.bmm(query, key.permute(0, 2, 1)) * scale  # ~(B, N, N)
        attention = similarity.softmax(dim=-1)
        self.attention_map = attention

        # attention * values
        output = torch.bmm(attention, value)
        return output

    def split_into_heads(self, x):
        """
        Splitting a vector into multiple heads
        """
        batch_size, num_tokens, _ = x.shape
        x = x.view(batch_size, num_tokens, self.num_heads, self.head_dim)  # split dimension into heads
        y = x.permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, num_tokens, self.head_dim)  # why?
        return y

    def merge_heads(self, x):
        """
        Rearranging heads and recovering original shape
        """
        _, num_tokens, dim_head = x.shape
        x = x.reshape(-1, self.num_heads, num_tokens, dim_head).transpose(1, 2)
        y = x.reshape(-1, num_tokens, self.num_heads * dim_head)
        return y


    def forward(self, x):
        """ 
        Forward pass through Self-Attention module
        """
        # linear projections and splitting into heads:
        # (B, N, D) --> (B, N, Nh, Dh) --> (B * Nh, N, Dh)
        q, k, v = self.q(x), self.k(x), self.v(x)
        q = self.split_into_heads(q)
        k = self.split_into_heads(k)
        v = self.split_into_heads(v)

        # applying attention equation
        vect = self.attention(query=q, key=k, value=v)

        # rearranging heads and recovering shape:
        # (B * Nh, N, Dh) --> (B N, Nh, Dh) --> (B, N, D)
        y = self.merge_heads(vect)
        y = self.out_proj(y)
        return y