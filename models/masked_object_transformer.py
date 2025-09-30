import torch
import torch.nn as nn
from models.encoder import SlotTransformerEncoder
from models.decoder import SlotTransformerDecoder

class MaskedObjectTransformer(nn.Module):
    def __init__(self, obj_num=10, embed_dim=256, img_size=64,
                 enc_depth=3, nhead=8, mlp_ratio=4.0):
        super().__init__()
        self.obj_num = obj_num
        self.img_size = img_size
        self.encoder = SlotTransformerEncoder(embed_dim=embed_dim, depth=enc_depth,
                                                nhead=nhead, mlp_ratio=mlp_ratio, num_slots=obj_num)
        self.decoder = SlotTransformerDecoder(embed_dim=embed_dim, img_size=img_size, obj_num=obj_num)

    def forward(self, item):
        """
          item: (imgs, masks)
          imgs: [B, 3, H, W]
          masks: [B, 1, H, W]
        """
        mem = self.encoder(item)
        recon = self.decoder(mem)

        return recon
