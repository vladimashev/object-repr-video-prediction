"""
1st (outdated) implementation for mask encoding reconstruction. To be deleted.
"""

import torch
import torch.nn as nn

from models.decoder import SlotTransformerDecoder

DEFAULT_OBJECT_NUM = 10

# =====================
# ObjectComposer and Slicer which create object by mask x image
# =====================

class ObjectSlicer(nn.Module):
    """Takes full images and instance-index masks and returns per-slot masked images.
       Output shape: [B, num_slots, C, H, W]
    """
    def __init__(self, obj_num: int = DEFAULT_OBJECT_NUM):
        super().__init__()
        self.obj_num = obj_num

    def forward(self, imgs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        # imgs: [B,C,H,W], masks: [B,H,W],
        # integer labels 0..obj_num-1 (background should be a label too if used)
        out = []
        for k in range(self.obj_num):
            mask_k = (masks == k).unsqueeze(1)            # [B,1,H,W]
            obj_k = imgs * mask_k.float()                 # [B,C,H,W]
            out.append(obj_k.unsqueeze(1))
        return torch.cat(out, dim=1)  # [B, K, C, H, W]

class ObjectComposer(nn.Module):
    """Composes per-slot RGB predictions back into a frame using the provided mask labels."""
    def forward(self, objects_rgb: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        # objects_rgb: [B, K, C, H, W], masks: [B,H,W]
        B, K, C, H, W = objects_rgb.shape
        recon = torch.zeros((B, C, H, W), device=objects_rgb.device, dtype=objects_rgb.dtype)
        for k in range(K):
            recon += objects_rgb[:,k] * (masks == k).unsqueeze(1).float()
        return recon

# =====================
# Encoder
# =====================

class ObjectCNNEncoder(nn.Module):
    """ Encodes a masked object image (3x64x64) into a single vector """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), # 4x4
            nn.ReLU(),
        )
        self.fc = nn.Linear(256*4*4, embed_dim)

    def forward(self, x):
        feat = self.conv(x)
        z = self.fc(feat.view(feat.size(0), -1))
        return z

class SlotTransformerEncoder(nn.Module):
    """ Transformer across object slots (self-attention among slot embeddings) """
    def __init__(self, embed_dim=256, depth=3, nhead=8, mlp_ratio=4.0, num_slots=10):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=int(embed_dim*mlp_ratio), batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        # optional learnable slot positional embeddings (helps the transformer distinguish slots)
        self.slot_pos = nn.Parameter(torch.randn(1, num_slots, embed_dim) * 0.02)

    def forward(self, slot_embeddings):
        # slot_embeddings: [B, K, D]
        x = slot_embeddings + self.slot_pos[:, :slot_embeddings.size(1), :]
        x = self.encoder(x)
        return x  # [B, K, D]

# =====================
# Decoder
# =====================

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

# =====================
# The reconstruction model
# =====================

class MaskedObjectTransformer(nn.Module):
    def __init__(self, obj_num=10, embed_dim=256, img_size=64, patch_size=8,
                 enc_depth=3, dec_depth=3, nhead=8, mlp_ratio=4.0):
        super().__init__()
        self.obj_num = obj_num
        self.img_size = img_size
        self.patch_size = patch_size
        self.encoder_cnn = ObjectCNNEncoder(embed_dim=embed_dim)
        self.slot_encoder = SlotTransformerEncoder(embed_dim=embed_dim, depth=enc_depth, nhead=nhead, mlp_ratio=mlp_ratio, num_slots=obj_num)
        self.slot_decoder = SlotTransformerDecoder(embed_dim=embed_dim, depth=dec_depth, nhead=nhead, mlp_ratio=mlp_ratio, img_size=img_size, patch_size=patch_size)
        self.slicer = ObjectSlicer(obj_num=obj_num)
        self.composer = ObjectComposer()

    def forward(self, item):
        imgs, masks = item
        # imgs: [B,3,H,W], masks: [B,H,W]
        B = imgs.size(0)
        objs = self.slicer(imgs, masks)  # [Batch_size,K_objects,3,Height,Width]
        B,K,C,H,W = objs.shape

        objs_flat = objs.view(B*K, C, H, W)
        z = self.encoder_cnn(objs_flat)          # [B*K, D]
        z_slots = z.view(B, K, -1)               # [B,K,D]

        # transformer across slots
        z_slots_refined = self.slot_encoder(z_slots)  # [B,K,D]

        # prepare for decoder
        mem = z_slots_refined.view(B*K, 1, -1)        # [B*K,1,D]
        # use transformer decoder with learned spatial queries
        obj_imgs = self.slot_decoder(mem)             # [B*K,3,H,W]
        obj_imgs = obj_imgs.view(B, K, 3, H, W)

        recon = self.composer(obj_imgs, masks)

        return z_slots_refined, recon, obj_imgs
