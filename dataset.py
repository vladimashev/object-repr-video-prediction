import os
import glob
import torch
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class MOViC_Dataset(Dataset):
    def __init__(self, root_dir, input_frames=5, target_frames=5, transform=None):
        self.root_dir = root_dir
        self.input_frames = input_frames
        self.target_frames = target_frames
        self.transform = transform

        rgb_files = sorted(glob.glob(os.path.join(root_dir, "rgb_*.png")))

        self.video_dict = {}
        for file in rgb_files:
            basename = os.path.basename(file)
            _, video_id, _ = basename.replace('.png', '').split('_')
            if video_id not in self.video_dict:
                self.video_dict[video_id] = []
            self.video_dict[video_id].append(file)

        self.sequences = []
        for video_id, frames in self.video_dict.items():
            total = len(frames)
            max_start = total - (input_frames + target_frames)
            if max_start < 0:
                continue
            #seed frames mean random? what for validation data?
            start_idx = random.randint(0, max_start)
            input_paths = frames[start_idx : start_idx + input_frames]
            target_paths = frames[start_idx + input_frames : start_idx + input_frames + target_frames]
            self.sequences.append((input_paths, target_paths))
            
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_paths, target_paths = self.sequences[idx]
        
        input_frames = [self.load_image(path) for path in input_paths]
        target_frames = [self.load_image(path) for path in target_paths]
        
        input_frames = torch.stack(input_frames)   # [input_frames, C, H, W]
        target_frames = torch.stack(target_frames) # [target_frames, C, H, W]
        
        return input_frames, target_frames

    def load_image(self, path):
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = np.array(img).astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1) # to C,H,W
        return img