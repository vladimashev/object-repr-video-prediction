import torch
import torch.nn as nn

DEFAULT_OBJECT_NUM = 10

class ObjectEncoder(nn.Module):
    def __init__(self, obj_num=DEFAULT_OBJECT_NUM):
        super().__init__()
        self.obj_num = obj_num

    def forward(self, item):
        imgs, masks = item    # imgs: [B, C, H, W], masks: [B, H, W]

        objects = []
        for k in range(self.obj_num):
            mask_k = (masks == k).unsqueeze(1)    # [B,1,H,W]
            obj_k = imgs * mask_k                 # [B,C,H,W]
            objects.append(obj_k.unsqueeze(1))    # keep slot dim
        objects = torch.cat(objects, dim=1)       # [B, obj_num, C, H, W]

        return objects  # [B, obj_num, C, H, W]


class ObjectDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, item):
        objects, masks = item
        recon = torch.zeros_like(objects[:,0])
        for k in range(objects.shape[1]):
            recon += objects[:,k] * (masks == k).unsqueeze(1).float()
        return recon

'''
class SlotEncoder(nn.Module):
    def __init__(self, in_ch=3, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.ReLU(True),
        )
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, d_model)
        )

    def forward(self, x):  # [B*object_num, C, H, W]
        f = self.net(x)
        z = self.proj(f)  # [B*object_num, d_model]
        return z

class SlotDecoder(nn.Module):
    def __init__(self, d_model=128, out_ch=3, out_size=(64,64)):
        super().__init__()
        H, W = out_size
        self.init_hw = (H // 4, W // 4)   # adjust to match encoder downsampling
        self.fc = nn.Linear(d_model, 128 * self.init_hw[0] * self.init_hw[1])
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(64, out_ch, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):  # [B*object_num, d_model]
        Bn, _D = z.shape
        h, w = self.init_hw
        x = self.fc(z).view(Bn, 128, h, w)
        return self.net(x)  # [B*object_num, C, H, W]

class ObjectAutoencoder(nn.Module):
    def __init__(self, object_encoder, object_decoder, obj_num=DEFAULT_OBJECT_NUM,
                 img_size=(64,64), in_ch=3, d_model=128):
        super().__init__()
        self.object_encoder = object_encoder
        self.object_decoder = object_decoder
        self.slot_enc = SlotEncoder(in_ch=in_ch, d_model=d_model)
        self.slot_dec = SlotDecoder(d_model=d_model, out_ch=in_ch, out_size=img_size)
        self.obj_num = obj_num
        self.d_model = d_model

    def forward(self, item):
        """
        frame:  [B,C,H,W]
        segmap: [B,H,W]   (object segmentation map)
        """
        frame, segmap = item
        # 1) Decompose frame into object slots
        slots = self.object_encoder((frame, segmap))  # [B, N, C, H, W]

        B, N, C, H, W = slots.shape
        # 2) Encode slots -> embeddings
        z = self.slot_enc(slots.reshape(B*N, C, H, W))         # [B*N, D]
        # 3) Decode back to slots
        slots_recon = self.slot_dec(z).reshape(B, N, C, H, W)  # [B, N, C, H, W]
        # 4) Reassemble frame
        frame_recon = self.object_decoder((slots_recon, segmap))      # [B, C, H, W]

        return frame_recon, slots, slots_recon
'''
