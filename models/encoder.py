import torch
import torch.nn as nn
from models.positional_encoding import PositionalEncoding
from models.transformer_block import TransformerBlock

class ViTPatchEncoder(nn.Module):
    """
    Encodes patches into embeddings.
    """
    def __init__(self, patch_dim, embed_dim, max_len, attn_dim, num_heads, mlp_size, num_tf_layers):
        super().__init__()
        self.patch_projection = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_dim)
        )
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

    def forward(self, x):
        """
        x:  (B, T*num_patches, patch_dim)
        Returns: (B, T*num_patches, embed_dim)
        """
        tokens = self.patch_projection(x)   # (B, L, embed_dim)
        tokens = self.pos_emb(tokens)       #
        out = self.transformer_blocks(tokens)  # (B, L, embed_dim)
        return out


class ConvPatchEncoder(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),   # 64x64 → 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 32x32 → 16x16
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),# 16x16 → 8x8
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),# 8x8 → 4x4
            nn.ReLU()
        )

    def forward(self, patches):
        # patches: (B, L, patch_dim) -> (B*L, 3, p, p)
        B, L, _ = patches.shape
        patches = patches.view(B * L, 3, self.patch_size, self.patch_size)
        out = self.encoder(patches)
        return out