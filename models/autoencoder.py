import torch.nn as nn
from models/patchifier import Patchifier
from torch import Tensor
# (input - kernel_size + 2*padding) / stride + 1


class ImageAutoencoder(nn.Module):
    def __init__(self, patch_size: int, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.patchifier = Patchifier(patch_size)
        self.encoder = encoder
        self.decoder = decoder
        self.patch_size = patch_size

    def forward(self,  x: Tensor)-> Tensor:  # x is of shape [B, T, C, H, W]
        patches = self.patchifier(x)  # [B, T*N_patches, patch_dim]
        batch_size, num_patches, patch_dim = patches.shape
        C = x.shape[2]
        patches = patches.view(batch_size * num_patches, C, self.patch_size, self.patch_size)  # [B*N_patches, 3, p, p]
        
        z = self.encoder(patches)
        recon = self.decoder(z)
        recon = recon.view(batch_size, num_patches, patch_dim)
        return recon

'''
class ImageAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder: 64x64 → 4x4
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),   # 64x64 → 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 32x32 → 16x16
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),# 16x16 → 8x8
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),# 8x8 → 4x4
            nn.ReLU()
        )

        # Decoder: 4x4 → 64x64
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), # 4x4 → 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 8x8 → 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 16x16 → 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),    # 32x32 → 64x64
            nn.Sigmoid()  # [0,1] output
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
'''