import torch.nn as nn
from models.encoder import ObjectCNNEncoder, SlotTransformerEncoder
from models.decoder import SlotTransformerDecoder
from models.object_composer import ObjectComposer, ObjectSlicer

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
