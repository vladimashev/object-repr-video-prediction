import torch
import torch.nn as nn
from models.encoder import SlotTransformerEncoder, DEFAULT_OBJECT_NUM
from models.decoder import SlotTransformerDecoder

class MaskedObjectTransformer(nn.Module):
    def __init__(self, obj_num=DEFAULT_OBJECT_NUM, embed_dim=256, img_size=64, patch_size=8,
                 enc_depth=3, dec_depth=3, nhead=8, mlp_ratio=4.0):
        super().__init__()
        self.obj_num = obj_num
        self.img_size = img_size
        self.encoder = SlotTransformerEncoder(embed_dim=embed_dim, depth=enc_depth,
                                                nhead=nhead, mlp_ratio=mlp_ratio, num_slots=obj_num)
        self.decoder = SlotTransformerDecoder(embed_dim=embed_dim, depth=dec_depth, nhead=nhead, obj_num=obj_num,
                                                mlp_ratio=mlp_ratio, img_size=img_size, patch_size=patch_size)

    def forward(self, item):
        mem = self.encoder(item)
        recon = self.decoder(mem)
        # obj_rgbs, obj_masks = self.decoder(mem)
        # imgs = item[0]
        # B = imgs.size(0)
        # K = self.obj_num
        # H = self.img_size
        # W = self.img_size
        # obj_rgbs = obj_rgbs.view(B,K,3,H,W)
        # obj_masks = obj_masks.view(B,K,1,H,W)
        # attn = obj_masks / (obj_masks.sum(dim=1, keepdim=True) + 1e-6)
        # recon = torch.sum(attn * obj_rgbs, dim=1)

        return recon
