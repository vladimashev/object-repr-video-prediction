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

class SlotTransformerDecoder(nn.Module):
    """ Decodes each slot embedding into an object image """
    def __init__(self, embed_dim=256, depth=3, nhead=8,
                 mlp_ratio=4.0, img_size=64, patch_size=8):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.patch_dim = 3 * patch_size * patch_size

        self.queries = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=int(embed_dim*mlp_ratio), batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        # head to predict raw patch pixels
        self.head = nn.Linear(embed_dim, self.patch_dim)

    def forward(self, slot_memory):
        # slot_memory: [B * num_slots, 1, D]
        Bk = slot_memory.size(0)
        q = self.queries.expand(Bk, -1, -1)  # [B*K, num_patches, D]
        # decoder expects memory: [B*K, M, D], here M=1
        out = self.decoder(tgt=q, memory=slot_memory)  # [B*K, num_patches, D]
        patches = self.head(out)  # [B*K, num_patches, patch_dim]
        
        patches = patches.view(Bk, self.grid, self.grid, 3, self.patch_size, self.patch_size)
        patches = patches.permute(0,3,1,4,2,5).contiguous()  # [B*K,3,grid,ps,grid,ps]
        obj_imgs = patches.view(Bk, 3, self.img_size, self.img_size)
        return obj_imgs
