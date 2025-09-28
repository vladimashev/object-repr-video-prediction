import os
import glob
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter, Resize, InterpolationMode, ToTensor

class SynchronizedTransform:
    def __init__(self, transform):
        """
        transform — torchvision.transforms.Compose([...])
        """
        self.transform = transform

    def __call__(self, frames: torch.Tensor, masks: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Parameters for random transformations for all frames
        do_hflip = False
        do_vflip = False
        rotation_angle = 0
        jitter_tf = None
        unsqueezed = False
        
        if frames.ndim != 4:
            unsqueezed = True
            frames, masks = frames.unsqueeze(0), masks.unsqueeze(0)

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

        # Apply the same augmentations for frames amd masks
        for idx, frame in enumerate(frames):
            for tf in self.transform.transforms:
                if isinstance(tf, RandomHorizontalFlip):
                    if do_hflip:
                        frame = F.hflip(frame)
                elif isinstance(tf, RandomVerticalFlip):
                    if do_vflip:
                        frame = F.vflip(frame)
                elif isinstance(tf, RandomRotation):
                    frame = F.rotate(frame, angle=rotation_angle, interpolation=InterpolationMode.BILINEAR)
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
            # Update the frame
            frames[idx] = frame

        if masks is not None:
            for idx, mask in enumerate(masks):
                for tf in self.transform.transforms:
                    if isinstance(tf, RandomHorizontalFlip):
                        if do_hflip:
                            mask = F.hflip(mask)
                    elif isinstance(tf, RandomVerticalFlip):
                        if do_vflip:
                            mask = F.vflip(mask)
                    elif isinstance(tf, RandomRotation):
                        mask = mask.unsqueeze(0) # [1, H, W]
                        mask = F.rotate(mask, angle=rotation_angle, interpolation=InterpolationMode.NEAREST)
                        mask = mask.squeeze(0) # back to [H, W]
                # Update the frame
                masks[idx] = mask
            
        if not unsqueezed:
            return frames, masks
        else:
            return frames.squeeze(0), masks.squeeze(0)


class MOViC_Dataset(Dataset):
    def __init__(self, root_dir, split="train", target='rgb', img_size = (64, 64), input_frames=5, target_frames=5, transform=None):
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
        self.masks = []
        
        #load frame paths
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

        #load masks paths
        if self.target == 'objects':
            mask_files = sorted(glob.glob(os.path.join(self.root_dir, "mask_*.pt")))
            
            for file in mask_files:
                mask_data = torch.load(file, map_location='cpu')
                masks = list(mask_data["masks"])
                self.masks.append(masks)
    
    def __len__(self):
        return len(self.sequences)

    # def __getitem__(self, idx):
    #     frame_paths = self.sequences[idx]
    #     mask_paths = self.masks[idx] if self.target == 'objects' else None

    #     if self.split == 'train': # subsample when training
    #         total = len(frame_paths)
    #         max_start = total - (self.input_frames + self.target_frames)
    #         start_idx = random.randint(0, max_start)
    #     elif self.split == 'validation': # no subsampling
    #         start_idx = 0

    #     frames, masks = None, None
    #     paths = frame_paths[start_idx : start_idx + self.input_frames + self.target_frames]
    #     frames = torch.stack([self._load_image(path) for path in paths])  # [T, C, H, W]
    #     frames = self.resizer_rgb(frames)
            
    #     if self.target == 'objects': # masks are stored as tensors
    #         masks = torch.stack(mask_paths[start_idx : start_idx + self.input_frames + self.target_frames])  # [T, C, H, W]
    #         masks = self.resizer_mask(masks)

    #     if self.split == 'train':
    #         frames, masks = self.transform(frames, masks)

    #     if masks is None:
    #         return frames
    #     else:
    #         # list of form [(frame, mask), ...]
    #         return [(f, m) for f, m in zip(frames, masks)]

            
    def __getitem__(self, idx):
        frame_paths = self.sequences[idx]
        mask_paths = self.masks[idx] if self.target == 'objects' else None
    
        # параметры выборки
        step = 2 if self.split == 'train' else 1
        need = self.input_frames + self.target_frames
        total = len(frame_paths)
    
        if self.split == 'train':  # subsample when training, with step
            # последний индекс последовательности: start_idx + step*(need-1)
            max_start = total - 1 - step * (need - 1)
            if max_start < 0:
                raise ValueError(
                    f"Sequence too short: total={total}, need={need}, step={step}"
                )
            start_idx = random.randint(0, max_start)
        elif self.split == 'validation':  # no subsampling
            start_idx = 0
    
        # индексы кадров с заданным шагом
        idxs = list(range(start_idx, start_idx + step * need, step))
    
        # загрузка кадров
        frames = torch.stack([self._load_image(frame_paths[i]) for i in idxs])  # [T, C, H, W]
        frames = self.resizer_rgb(frames)
    
        # загрузка масок (хранятся как тензоры) тем же шагом
        masks = None
        if self.target == 'objects':
            masks = torch.stack([mask_paths[i] for i in idxs])  # [T, C, H, W]
            masks = self.resizer_mask(masks)
    
        # аугментации только на train
        if self.split == 'train':
            frames, masks = self.transform(frames, masks)
    
        # формат выхода
        if masks is None:
            return frames
        else:
            #return [(f, m) for f, m in zip(frames, masks)]
            return frames, masks
    
    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1) # to C,H,W
        return img


# ================================= FLATTENED DATASETS ================================ 

DEFAULT_IMAGE_SIZE = (64, 64)

class FrameDataset(Dataset):
    """ Abstract class for flattened mask/frame storage, no aggregation by sequence """
    def __init__(self, image_dir, transform, img_size = DEFAULT_IMAGE_SIZE):
        super().__init__()
        self.image_dir = image_dir
        self.transform = SynchronizedTransform(transform)
        self.resizer_rgb = Resize(
                img_size,
                interpolation=InterpolationMode.BILINEAR
            )
        self.resizer_mask = Resize(
                img_size,
                interpolation=InterpolationMode.NEAREST
            )
        self.paths_rgb = []
        self._load_data()

    def __len__(self):
        return len(self.paths_rgb)

    def _load_data(self):
        for filepath in sorted(glob.glob(os.path.join(self.image_dir, "rgb_*.png"))):
            self.paths_rgb.append(os.path.basename(filepath))
            
        return
    
    def __getitem__(self, idx):
        img_path = self.paths_rgb[idx]
        img = Image.open(os.path.join(self.image_dir, img_path)).convert("RGB")
        img = self.resizer_rgb(img)
        img = ToTensor()(img) # Tensor [3, H, W]


        _, video_id, frame_id = img_path.replace('.png', '').split('_')
        mask_path = os.path.join(self.image_dir, f"mask_{video_id}.pt")
        mask_data = torch.load(mask_path, map_location='cpu')
        img_mask = mask_data["masks"][int(frame_id)]
        mask_tensor = torch.from_numpy(np.array(img_mask)).long().unsqueeze(0) # Tensor [1, H, W]
        img_mask = self.resizer_mask(mask_tensor)

        # transform synchronically
        img_tensor, mask_tensor = self.transform(img, img_mask)

        return {
            "img": img_tensor,
            "mask": mask_tensor,
            "video_id": int(video_id),
            "frame_id": int(frame_id)
        }

class ImageDataset(FrameDataset):
    """ Full frame dataset """
    def __init__(self, image_dir, transform, img_size = DEFAULT_IMAGE_SIZE):
        super().__init__(image_dir, transform, img_size)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        
        return item["img"].unsqueeze(0) # add T dim

class MaskDataset(FrameDataset):
    """ Masked frame dataset.
        Returns (img, mask) tuples. """
    def __init__(self, image_dir, transform, img_size = DEFAULT_IMAGE_SIZE):
        super().__init__(image_dir, transform, img_size)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        
        # add T dim for images to match predictor's format
        return item["img"].unsqueeze(0), item["mask"]#.unsqueeze(0)
    