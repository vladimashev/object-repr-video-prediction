import os
import glob
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter, Resize, InterpolationMode


class SynchronizedTransform:
    def __init__(self, transform):
        """
        transform — torchvision.transforms.Compose([...])
        """
        self.transform = transform

    def __call__(self, input_frames: torch.Tensor, target_frames: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # Parameters for random transformations for all frames
        do_hflip = False
        do_vflip = False
        rotation_angle = 0
        jitter_tf = None

        # Find parameters
        for tf in self.transform.transforms:
            if isinstance(tf, RandomHorizontalFlip):
                do_hflip = random.random() < tf.p
            elif isinstance(tf, RandomVerticalFlip):
                do_vflip = random.random() < tf.p
            elif isinstance(tf, RandomRotation):
                rotation_angle = tf.get_params(tf.degrees)
            elif isinstance(tf, ColorJitter):
                jitter_tf = tf.get_params(
                    tf.brightness, tf.contrast, tf.saturation, tf.hue
                )

        # Apply the same augmentations for input and target frames
        for idx, frame in enumerate(input_frames):
            for tf in self.transform.transforms:
                if isinstance(tf, RandomHorizontalFlip):
                    if do_hflip:
                        frame = F.hflip(frame)
                elif isinstance(tf, RandomVerticalFlip):
                    if do_vflip:
                        frame = F.vflip(frame)
                elif isinstance(tf, RandomRotation):
                    frame = F.rotate(frame, angle=rotation_angle)
                elif isinstance(tf, ColorJitter):
                    order, b, c, s, h = jitter_tf
                    for transform_idx in order.tolist():
                        if transform_idx == 0 and b is not None: # Adjust brightness
                            frame = F.adjust_brightness(frame, b)
                        elif transform_idx == 1 and c is not None: # Adjust contrast
                            frame = F.adjust_contrast(frame, c) 
                        elif transform_idx == 2 and s is not None: # Adjust saturation
                            frame = F.adjust_saturation(frame, s) 
                        elif transform_idx == 3 and h is not None: # Adjust hue
                            frame = F.adjust_hue(frame, h)
                else:
                    frame = tf(frame)
            # Update the frame
            input_frames[idx] = frame
            
        for idx, frame in enumerate(target_frames):
            for tf in self.transform.transforms:
                if isinstance(tf, RandomHorizontalFlip):
                    if do_hflip:
                        frame = F.hflip(frame)
                elif isinstance(tf, RandomVerticalFlip):
                    if do_vflip:
                        frame = F.vflip(frame)
                elif isinstance(tf, RandomRotation):
                    frame = F.rotate(frame, angle=rotation_angle)
                elif isinstance(tf, ColorJitter):
                    order, b, c, s, h = jitter_tf
                    for transform_idx in order.tolist():
                        if transform_idx == 0 and b is not None: # Adjust brightness
                            frame = F.adjust_brightness(frame, b)
                        elif transform_idx == 1 and c is not None: # Adjust contrast
                            frame = F.adjust_contrast(frame, c) 
                        elif transform_idx == 2 and s is not None: # Adjust saturation
                            frame = F.adjust_saturation(frame, s) 
                        elif transform_idx == 3 and h is not None: # Adjust hue
                            frame = F.adjust_hue(frame, h)
                else:
                    frame = tf(frame)
            # Update the frame
            target_frames[idx] = frame
            
        return input_frames, target_frames


class MOViC_Dataset(Dataset):
    def __init__(self, root_dir, split="train", target='mask', img_size = (64, 64), input_frames=5, target_frames=5, transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.split = split
        self.target = target
        self.img_size = img_size
        self.input_frames = input_frames
        self.target_frames = target_frames
        self.transform = SynchronizedTransform(transform)

        self.resizer_rgb = Resize(
                self.img_size,
                interpolation=InterpolationMode.BILINEAR
            )
        self.resizer_mask = Resize(
                self.img_size,
                interpolation=InterpolationMode.NEAREST
            )
        
        self.sequences = []
        if self.target == 'rgb':
            current_video = None
            current_sequence = []
            
            for file in sorted(glob.glob(os.path.join(self.root_dir, "rgb_*.png"))):
                basename = os.path.basename(file)
                _, video_id, _ = basename.replace('.png', '').split('_')
            
                if video_id != current_video:
                    if current_sequence:
                        self.sequences.append(current_sequence)
                    current_sequence = [file]
                    current_video = video_id
                else:
                    current_sequence.append(file)
                    
            # add the last remaining sequence
            if current_sequence:
                self.sequences.append(current_sequence)
        elif self.target == 'mask':
            mask_files = sorted(glob.glob(os.path.join(self.root_dir, "mask_*.pt")))
            
            for file in mask_files:
                mask_data = torch.load(file, map_location='cpu')
                masks = list(mask_data["masks"])
                self.sequences.append(masks)
    
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        frame_paths = self.sequences[idx]

        if self.split == 'train': # subsample when training
            total = len(frame_paths)
            max_start = total - (self.input_frames + self.target_frames)
            start_idx = random.randint(0, max_start)
        elif self.split == 'validation': # no subsampling
            start_idx = 0
            
        if self.target == 'rgb':
            input_paths = frame_paths[start_idx : start_idx + self.input_frames]
            target_paths = frame_paths[start_idx + self.input_frames : start_idx + self.input_frames + self.target_frames]
            input_frames = torch.stack([self._load_image(path) for path in input_paths])  # [input_frames, C, H, W]
            target_frames = torch.stack([self._load_image(path) for path in target_paths])  # [input_frames, C, H, W]

            input_frames, target_frames = self.resizer_rgb(input_frames), self.resizer_rgb(target_frames)
            
        elif self.target == 'mask': # they are stored as tensors
            input_frames = torch.stack(frame_paths[start_idx : start_idx + self.input_frames]) # [input_frames, C, H, W]
            target_frames = torch.stack(frame_paths[start_idx + self.input_frames : start_idx + self.input_frames + self.target_frames])  # [input_frames, C, H, W]
            input_frames, target_frames = self.resizer_mask(input_frames), self.resizer_mask(target_frames)
        print(input_frames.shape, target_frames.shape)
        if self.split == 'train':
            input_frames, target_frames = self.transform(input_frames, target_frames)
        
        return input_frames, target_frames

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1) # to C,H,W
        return img