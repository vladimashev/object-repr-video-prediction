import torch
import torch.nn as nn

DEFAULT_OBJECT_NUM = 10

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
