import torch
import torch.nn as nn
from models.multi_head_self_attention import MultiHeadSelfAttention
from models.mlp import MLP
from models.positional_encoding import PositionalEncoding, PositionalEncoding2D

class TransformerBlock(nn.Module):
    """
    Transformer block using self-attention

    Args:
    -----
    token_dim: int
        Dimensionality of the input tokens
    attn_dim: int
        Inner dimensionality of the attention module. Must be divisible be num_heads
    num_heads: int
        Number of heads in the self-attention mechanism
    mlp_size: int
        Hidden dimension of the MLP module
    """

    def __init__(self, token_dim, attn_dim, num_heads, mlp_size, causal=False):
        """ Module initializer """
        super().__init__()
        self.token_dim = token_dim
        self.mlp_size = mlp_size
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.causal = causal

        # MHA
        self.ln_att = nn.LayerNorm(token_dim, eps=1e-6)
        self.attn = MultiHeadSelfAttention(
                token_dim=token_dim,
                attn_dim=attn_dim,
                num_heads=num_heads
            )
        
        # MLP
        self.ln_mlp = nn.LayerNorm(token_dim, eps=1e-6)
        self.mlp = MLP(
                in_dim=token_dim,
                hidden_dim=mlp_size,
            )
        return


    def forward(self, inputs, attn_mask=None):
        """
        Forward pass through transformer encoder block.
        We assume the more modern PreNorm design
        """
        assert inputs.ndim == 3

        # Self-attention.
        x = self.ln_att(inputs)

        attn_mask = None
        if self.causal:
            N = inputs.size(1)
            attn_mask = torch.triu(torch.ones(N, N, device=inputs.device), diagonal=1).bool()
        
        x = self.attn(x, attn_mask=attn_mask)
        y = x + inputs

        # MLP
        z = self.ln_mlp(y)
        z = self.mlp(z)
        z = z + y

        return z


    def get_attention_masks(self):
        """ Fetching last computer attention masks """
        attn_masks = self.attn.attention_map
        N = attn_masks.shape[-1]
        attn_masks = attn_masks.reshape(-1, self.num_heads, N, N)
        return attn_masks


class SpatialTemporalBlock(nn.Module):
    def __init__(self, token_dim, attn_dim, num_heads, mlp_size, grid, max_len, causal=False, target='rgb'):
        super().__init__()
        self.causal = causal
        self.spatial_pe = PositionalEncoding2D(token_dim, grid) if target=='rgb' else None
        self.temporal_pe = PositionalEncoding(token_dim, max_len)
        self.dropout = nn.Dropout(0.3)

        # --- Spatial subblock ---
        self.ln_spatial = nn.LayerNorm(token_dim)
        self.attn_spatial = MultiHeadSelfAttention(token_dim, attn_dim, num_heads)
        self.ln_mlp_spatial = nn.LayerNorm(token_dim)
        self.mlp_spatial = MLP(token_dim, mlp_size)

        # --- Temporal subblock ---
        self.ln_temporal = nn.LayerNorm(token_dim)
        self.attn_temporal = MultiHeadSelfAttention(token_dim, attn_dim, num_heads)
        self.ln_mlp_temporal = nn.LayerNorm(token_dim)
        self.mlp_temporal = MLP(token_dim, mlp_size)

    @staticmethod
    def build_causal_mask(T: int, Np: int, device: torch.device):
        # один временной блок (T,T), без разнесения по патчам
        t = torch.arange(T, device=device)
        delta = t.unsqueeze(1) - t.unsqueeze(0)   # i - j
        allowed = (delta >= 0) & (delta <= 5)     # видеть только 5 предыдущих и себя
        time_mask = ~allowed                      # True = запретить
        return time_mask

    def forward(self, x, target='rgb', causal=False):
        """
        x: (B, T, Np, D)


        [1, 2,3,4,5,6,7,8,9]
                  _, _
        """
        B, T, Np, D = x.shape

        # === Spatial Attention within each frame ===
        x_spatial = x
        if target == 'rgb': # no PE in case of objects
            x_spatial = self.spatial_pe(x_spatial)
        x_spatial = x_spatial.reshape(B * T, Np, D) # [B*T, Np, D]
        xs = self.ln_spatial(x_spatial)
        xs = self.attn_spatial(xs)
        #xs = self.dropout(xs)
        xs = xs + x_spatial
        xs = xs + self.mlp_spatial(self.ln_mlp_spatial(xs))
        xs = xs.reshape(B, T*Np, D) # [B, T*Np, D]

        # === Temporal Attention between frames ===
        xt = self.temporal_pe(xs)
        xt = xt.reshape(B, T, Np, D).transpose(1, 2).reshape(B * Np, T, D)
        xt = self.ln_temporal(xt)
        mask = None
        if causal:
            mask = build_causal_mask(T, Np, xt.device)
        xt = self.attn_temporal(xt, attn_mask=mask)
        #xt = self.dropout(xt)
        xt = xt.reshape(B, Np, T, D).transpose(1, 2).reshape(B, T*Np, D)
        xt = xt + xs
        xt = xt + self.mlp_temporal(self.ln_mlp_temporal(xt))

        xt = xt.reshape(B, T, Np, D)

        return xt