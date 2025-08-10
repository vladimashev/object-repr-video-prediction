import torch.nn as nn
from models.patchifier import Patchifier
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
