import torch
import torch.nn as nn
from models.encoder import ObjectCNNEncoder, SlotTransformerEncoder, DEFAULT_OBJECT_NUM
from models.decoder import SlotTransformerDecoder

class MaskedObjectTransformer(nn.Module):
    def __init__(self, obj_num=DEFAULT_OBJECT_NUM, embed_dim=256, img_size=64, patch_size=8,
                 enc_depth=3, dec_depth=3, nhead=8, mlp_ratio=4.0):
        super().__init__()
        self.obj_num = obj_num
        self.encoder_cnn = ObjectCNNEncoder(embed_dim=embed_dim)
        self.slot_encoder = SlotTransformerEncoder(embed_dim=embed_dim, depth=enc_depth,
                                                   nhead=nhead, mlp_ratio=mlp_ratio, num_slots=obj_num)
        self.slot_decoder = SlotTransformerDecoder(embed_dim=embed_dim, depth=dec_depth, nhead=nhead,
                                                   mlp_ratio=mlp_ratio, img_size=img_size, patch_size=patch_size)

    def forward(self, item):
        imgs, masks = item
        
        B = imgs.size(0)
        # slice objects
        objs = []
        for k in range(self.obj_num):
            mask_k = (masks == k).unsqueeze(1)
            objs.append(imgs * mask_k.float())
        objs = torch.stack(objs, dim=1)
        B,K,C,H,W = objs.shape

        z = self.encoder_cnn(objs.view(B*K,C,H,W))
        z_slots = z.view(B, K, -1)
        z_slots_refined = self.slot_encoder(z_slots)

        mem = z_slots_refined.view(B*K, 1, -1)
        obj_rgbs, obj_masks = self.slot_decoder(mem)
        obj_rgbs = obj_rgbs.view(B,K,3,H,W)
        obj_masks = obj_masks.view(B,K,1,H,W)

        attn = obj_masks / (obj_masks.sum(dim=1, keepdim=True) + 1e-6)
        recon = torch.sum(attn * obj_rgbs, dim=1)

        return z_slots_refined, recon, obj_rgbs, obj_masks
