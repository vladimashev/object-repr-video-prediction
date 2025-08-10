import torch
import torch.nn as nn
from models.positional_encoding import PositionalEncoding
from models.transformer_block import TransformerBlock


class ViTPatchDecoder(nn.Module):
    """
    Decodes emdebbings back to patches
    """
    def __init__(self, patch_size, embed_dim, max_len, attn_dim, num_heads, mlp_size, num_tf_layers,
                 use_positional_encoding: bool = True):
        super().__init__()

        self.input_norm = nn.LayerNorm(embed_dim)
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.pos_emb = PositionalEncoding(embed_dim, max_len)

        blocks = [
            TransformerBlock(
                token_dim=embed_dim,
                attn_dim=attn_dim,
                num_heads=num_heads,
                mlp_size=mlp_size
            )
            for _ in range(num_tf_layers)
        ]
        self.transformer_blocks = nn.Sequential(*blocks)

        # project from embedding dimension to patches
        patch_dim = patch_size * patch_size * 3
        self.to_patch = nn.Linear(embed_dim, patch_dim)

    def forward(self, x):
        """
        x: (B, T*num_patches, embed_dim)
        -> (B, T*num_patches, patch_dim)
        Here T=1 actually, so sequence legnth=1 (because we apply attention only to the patches in the cuttent frame)
        """
        tokens = self.input_norm(x)

        if self.use_positional_encoding:
            tokens = self.pos_emb(tokens)

        tokens = self.transformer_blocks(tokens)  # (B, T*num_patches, embed_dim)
        out = self.to_patch(tokens)               # (B, T*num_patches, patch_dim)
        return out

