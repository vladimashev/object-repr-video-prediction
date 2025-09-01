import torch
import torch.nn as nn
from models.patchifier import Patchifier
from models.positional_encoding import PositionalEncoding
from models.transformer_block import TransformerBlock


class ViTPatchDecoder(nn.Module):
    """
    Decodes emdebbings back to patches
    """
    def __init__(self, H, W, patch_size, embed_dim, max_len, attn_dim, num_heads, mlp_size, num_tf_layers,
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

        self.H = H
        self.W = W
        self.patch_size = patch_size
        # project from embedding dimension to patches
        patch_dim = patch_size * patch_size * 3
        self.to_patch = nn.Linear(embed_dim, patch_dim)

    def forward(self, x):
        """
        x: (B, T, num_patches, embed_dim)
        -> (B, T, num_patches, patch_dim)
        """
        B, T, num_patches, embed_dim = x.shape
        tokens = self.input_norm(x)

        # per-frame attention: (B, T*num_patches, embed_dim) -> (B, T, num_patches, embed_dim) -> (B*T, num_patches, embed_dim)
        tokens = tokens.reshape(B * T, num_patches, embed_dim)
        if self.use_positional_encoding:
            tokens = self.pos_emb(tokens)

        tokens = self.transformer_blocks(tokens)  # (B*T, num_patches, embed_dim)
        out = self.to_patch(tokens)               # (B*T, num_patches, patch_dim)
        out = out.view(B, T, num_patches, -1) # (B, T, num_patches, patch_dim)

        nH, nW = self.H // self.patch_size, self.W // self.patch_size
        # (B, T, num_patches, patch_dim) -> (B,T,nH,nW,C,p,p) -> (B,T,C,H,W)
        out = out.view(B, T, nH, nW, 3, self.patch_size, self.patch_size).permute(0,1,4,2,5,3,6).contiguous()
        out = out.view(B, T, 3, self.H, self.W)
        
        return out

class ObjectTransformerDecoder(nn.Module):
    def __init__(self, obj_num, C, H, W, embed_dim, attn_dim, num_heads, mlp_size, num_tf_layers):
        super().__init__()
        self.obj_num = obj_num
        self.slot_dim = C * H * W
        self.C = C
        self.H = H
        self.W = W

        self.input_norm = nn.LayerNorm(embed_dim)
        self.pos_emb = PositionalEncoding(embed_dim, obj_num)

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

        self.to_object = nn.Linear(embed_dim, self.slot_dim)

    def forward(self, tokens):
        """
        tokens: [B, O, embed_dim]
        returns: [B, O, C, H, W] object reconstructions
        """
        B, O, _ = tokens.shape
        tokens = self.input_norm(tokens)
        tokens = self.pos_emb(tokens)
        tokens = self.transformer_blocks(tokens)   # [B, O, embed_dim]
        out = self.to_object(tokens)               # [B, O, slot_dim]
        out = out.view(B, O, self.C, self.H, self.W)             # [B, O, C, H, W]
        return out
