import torch.nn as nn
from models.patchifier import Patchifier
from torch import Tensor
# (input - kernel_size + 2*padding) / stride + 1


class ImageAutoencoder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, patch_size: int):
        super().__init__()
        self.patchifier = Patchifier(patch_size)
        self.encoder = encoder
        self.decoder = decoder
        self.patch_size = patch_size

    def forward(self,  x: Tensor)-> Tensor:  # x is of shape [B, C, H, W]
        x = x.unsqueeze(1) # x is of shape [B, T, C, H, W]
        B, T, C, H, W = x.shape
        
        patches = self.patchifier(x)  # [B, T*N_patches, patch_dim]

        batch_size, num_patches, patch_dim = patches.shape
        z = self.encoder(patches)
        recon = self.decoder(z)

        recon = recon.view(batch_size, num_patches, patch_dim)
        nH, nW = H // self.patch_size, W // self.patch_size
        # (B,T*N_patches, patch_dim) -> (B,T,nH,nW,C,p,p) -> (B,T,C,H,W)
        recon = recon.view(B, T, nH, nW, C, self.patch_size, self.patch_size).permute(0,1,4,2,5,3,6).contiguous()
        recon = recon.view(B, T, C, H, W)

        # remove T
        recon = recon.squeeze(1)
        return recon
