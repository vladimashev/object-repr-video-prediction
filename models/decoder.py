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
        Returns: (B, T, C, H, W)
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
    def __init__(self, obj_num = 10, embed_dim=256, img_size=64):
        super().__init__()
        self.obj_num = obj_num
        self.img_size = img_size
        self.fc = nn.Linear(embed_dim, 128*8*8)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 4, 3, padding=1) # 3 RGB + 1 mask
        )

    def forward(self, slots):
        B, T, K, D = slots.shape
        H, W = self.img_size, self.img_size
        x = self.fc(slots.view(B*T*K, D)).view(B*T*K, 128, 8, 8)

        out = self.deconv(x)

        # RGB
        rgb = torch.sigmoid(out[:, :3])   # [B*T*K, 3, H, W]
        obj_rgbs = rgb.view(B, T, K, 3, H, W)

        mask_logits = out[:, 3:4]
        # reshape to [B, T, K, H, W]
        mask_logits = mask_logits.view(B, T, K, 1, H, W)
        obj_masks = torch.softmax(mask_logits, dim=2)   # [B,T,K,1,H,W]

        recon = torch.sum(obj_masks * obj_rgbs, dim=2)   # [B,T,3,H,W]

        return recon, mask_logits.argmax(dim=2).squeeze(2), mask_logits.squeeze(3)
