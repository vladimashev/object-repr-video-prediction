from torch.utils.data import Dataset
from PIL import Image
import os
import glob
import torch

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform):
        super().__init__()
        self.image_dir = image_dir
        self.transform = transform
        self.paths = []
        self._load_data()

    def __len__(self):
        return len(self.paths)

    def _load_data(self):
        for filepath in sorted(glob.glob(os.path.join(self.image_dir, "rgb_*.png"))):
            self.paths.append(os.path.basename(filepath))
            
        return
    
    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(os.path.join(self.image_dir, img_path)).convert("RGB")

        _, video_id, frame_id = img_path.replace('.png', '').split('_')
        mask_path = os.path.join(self.image_dir, f"mask_{video_id}.pt")
        mask_data = torch.load(mask_path, map_location='cpu')
        img_mask = mask_data["masks"][int(frame_id)]

        return {
            "img": self.transform(img),
            "mask": img_mask
        }
