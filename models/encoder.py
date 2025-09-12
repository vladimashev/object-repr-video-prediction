import torch
import torch.nn as nn
from models.patchifier import Patchifier
from models.positional_encoding import PositionalEncoding
from models.transformer_block import TransformerBlock

class ViTPatchEncoder(nn.Module):
    """
    Breaks frames into patches and encodes into embeddings.
    """
    def __init__(self, patch_size, embed_dim, max_len, attn_dim, num_heads, mlp_size, num_tf_layers):
        super().__init__()
        self.patchifier = Patchifier(patch_size)
        self.patch_projection = nn.Sequential(
            nn.LayerNorm(patch_size * patch_size * 3),
            nn.Linear(patch_size * patch_size * 3, embed_dim)
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
        x:  (B, T, C, H, W)
        Returns: (B, T, num_patches, embed_dim)
        """
        B, T, _, _, _ = x.shape
        
        patches = self.patchifier(x)  # [B, T*num_patches, patch_dim]
        tokens = self.patch_projection(patches)   # (B, T*num_patches, embed_dim)

        embed_dim = tokens.size(-1)
        # per-frame attention: (B, T*num_patches, embed_dim) -> (B, T, num_patches, embed_dim) -> (B*T, num_patches, embed_dim)
        tokens = tokens.view(B, T, -1, embed_dim).reshape(B * T, -1, embed_dim)
        
        tokens = self.pos_emb(tokens)
        out = self.transformer_blocks(tokens)  # (B*T, num_patches, embed_dim)
        out = tokens.view(B, T, -1, embed_dim)   # (B, T, num_patches, embed_dim)
        return out

# =====================
# Masked encoder
# =====================

class ObjectCNNEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(256*4*4, embed_dim)

    def forward(self, x):
        feat = self.conv(x)
        z = self.fc(feat.view(feat.size(0), -1))
        return z

class SlotTransformerEncoder(nn.Module):
    def __init__(self, embed_dim=256, depth=3, nhead=8, mlp_ratio=4.0, num_slots=10):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead,
                                                   dim_feedforward=int(embed_dim*mlp_ratio), batch_first=True)
        self.encoder_cnn = ObjectCNNEncoder(embed_dim=embed_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.slot_pos = nn.Parameter(torch.randn(1, num_slots, embed_dim) * 0.02)
        self.num_slots = num_slots

    def forward(self, item):
        imgs, masks = item
        
        B = imgs.size(0)

        objs = []
        for k in range(self.num_slots):
            mask_k = (masks == k).unsqueeze(1)
            objs.append(imgs * mask_k.float())
        objs = torch.stack(objs, dim=1)
        B,K,C,H,W = objs.shape

        z = self.encoder_cnn(objs.view(B*K,C,H,W))
        slot_embeddings = z.view(B, K, -1)
        
        z_slots_refined = self.encoder(slot_embeddings)

        mem = z_slots_refined.view(B*K, -1)

        return mem
    