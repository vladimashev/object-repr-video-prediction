class Patchifier:
    """ 
    Splits video sequence (B, T, C, H, W) into non-overlapping 2D patches, where:
    B - batch size, 
    T - sequence length (number of frames), 
    C - number of channels, 
    and H and W are height and width respectively.
    
    Returns shape: (B, T * N_patches, patch_dim), where:
    N_patches - number of patches per frame,
    patch_dim - size of patches (square)
    """

    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, x):
        if x.ndim != 5:
            raise ValueError(f"Expected input of shape (B, T, C, H, W), got {x.shape}")
        
        B, T, C, H, W = x.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(f"Image size ({H}, {W}) must be divisible by patch size {self.patch_size}")

        num_patch_H = H // self.patch_size
        num_patch_W = W // self.patch_size
        num_patches_per_frame = num_patch_H * num_patch_W
        patch_dim = C * self.patch_size * self.patch_size

        x = x.view(B * T, C, H, W)  # [B*T, C, H, W]

        patches = x.reshape(B * T, C, num_patch_H, self.patch_size, num_patch_W, self.patch_size)
        patches = patches.permute(0, 2, 4, 1, 3, 5)  # [B*T, nH, nW, C, p, p]
        patches = patches.reshape(B, T * num_patches_per_frame, patch_dim)  # [B, T * N_patches, patch_dim]

        return patches