"""
Original source:
https://github.com/google-research/google-research/blob/master/frechet_video_distance/frechet_video_distance.py
"""

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from scipy import linalg


def preprocess_frames(frames, target_resolution=(224, 224), device=None):
    """
    Preprocess frames for a I3D-style encoder.

    Args:
      frames: torch.Tensor, shape [B, T, C, H, W].
      target_resolution: (height, width) tuple *or* (width, height) depending on usage.
                         We will interpret it as (H_out, W_out) here.
      device: torch device to put the returned tensor on.

    Returns:
      torch.FloatTensor of shape [B, T, C, H_out, W_out], dtype float32, values in [-1, 1].
    """
    assert frames.ndim == 5, "Expected shape [B, T, C, H, W]"

    B, T, C, H, W = frames.shape
    
    f = frames.clone().float()
    # Combine batch and time to apply interpolation once
    all_frames = f.view(B * T, C, H, W)  # [B*T, C, H, W]
    # Resize using bilinear
    H_out, W_out = target_resolution
    resized_videos = F.interpolate(all_frames, size=(H_out, W_out), mode='bilinear', align_corners=False)
    # reshape back
    output_videos = resized_videos.view(B, T, C, H_out, W_out)
    # scale to [-1, 1]
    scaled_videos = (2.0 * output_videos / 255.0) - 1.0
    
    if device is not None:
        scaled_videos = scaled_videos.to(device)
    
    return scaled_videos


class VideoEmbedder(torch.nn.Module):
    """
    Wrapper that produces a per-video embedding [B, feat_dim] from input
    frames [B, T, C, H, W].
    """

    def __init__(self, device='cpu', pretrained=True):
        super().__init__()
        self.device = torch.device(device)
        self.model = self._build_default_model(pretrained).to(self.device)
        self.model.eval()

    def _build_default_model(self, pretrained):
        model = torchvision.models.video.r3d_18(pretrained=pretrained)
        
        layers = torch.nn.Sequential(
            model.stem,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool  # results in shape [B, C_feat, 1, 1, 1] for video model avgpool
        )

        class FeatureExtractor(torch.nn.Module):
            def __init__(self, body, feat_dim):
                super().__init__()
                self.body = body
                self.feat_dim = feat_dim

            def forward(self, x):
                # x expected [B, C, T, H, W] for torchvision video models
                out = self.body(x)  # shape [B, C_feat, 1, 1, 1]
                out = out.view(out.size(0), -1)  # [B, feat_dim]
                return out

        feat_dim = model.fc.in_features
        
        return FeatureExtractor(layers, feat_dim)

    @torch.no_grad()
    def forward(self, frames):
        """
        Args:
          frames: torch.Tensor [B, T, C, H, W], preprocessed in range [-1,1]
        Returns:
          embeddings: torch.FloatTensor [B, feat_dim]
        """
        # torchvision video models expect [B, C, T, H, W]
        x = frames.permute(0, 2, 1, 3, 4).contiguous()
        x = x.to(self.device)
        out = self.model(x)
        
        return out


def get_activations_from_frames(frames, embedder, batch_size=16, device='cpu', target_resolution=(224,224)):
    """
    Compute embeddings for all frames.

    Args:
      frames: torch.Tensor or numpy array, shape [B, T, C, H, W]
      embedder: VideoEmbedder (or any nn.Module producing [B, D] embeddings)
      batch_size: int, how many frames to process at once
      device: device for computation
      target_resolution: resize target (H, W) required by the pretrained model

    Returns:
      activations: numpy array shape [B, D], dtype float64 (for stable cov computations)
    """
    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames)
    
    frames = frames.to(torch.device(device))
    B = frames.shape[0]
    acts = []
    
    for i in range(0, B, batch_size):
        batch = frames[i:i+batch_size]
        batch = preprocess_frames(batch, target_resolution=target_resolution, device=device)
        with torch.no_grad():
            emb = embedder(batch)  # [B, D]
        emb = emb.cpu().numpy()
        acts.append(emb)
    
    activations = np.concatenate(acts, axis=0)
    
    return activations


def get_frechet_distance(mu1, sigma1, mu2, sigma2):
    """
    Calculates the Frechet distance between two multivariate Gaussians.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    d = mu1.shape[0]
    assert mu2.shape[0] == d, "Mean vectors have different lengths"

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    tr_covmean = np.trace(covmean)
    fd = (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)
    
    return float(np.real(fd))


def calculate_fvd(real_acts, gen_acts):
    """
    Args:
      real_acts: numpy array [N, D]
      gen_acts: numpy array [N, D]  (or different N)

    Returns:
      scalar float (FVD)
    """
    real_mu = np.mean(real_acts, axis=0)
    gen_mu = np.mean(gen_acts, axis=0)
    real_sigma = np.cov(real_acts, rowvar=False)
    gen_sigma = np.cov(gen_acts, rowvar=False)
    
    return get_frechet_distance(real_mu, real_sigma, gen_mu, gen_sigma)


def execute_default_fvd(real_frames, gen_frames, device="cuda"):
    """ Initializes with a default model, gets activations and computes FVD """
    embedder = VideoEmbedder(device=device, pretrained=True)
    batch_size = real_frames.shape[0]
    real_acts = get_activations_from_frames(real_frames, embedder, batch_size=4, device=device, target_resolution=(224,224))
    gen_acts = get_activations_from_frames(gen_frames, embedder, batch_size=4, device=device, target_resolution=(224,224))

    fvd_value = calculate_fvd(real_acts, gen_acts)

    return fvd_value

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    B = 8
    T = 16
    C = 3
    H = 256
    W = 256
    
    real_frames = (np.random.rand(B, T, C, H, W) * 255).astype(np.uint8)
    gen_frames = (np.random.rand(B, T, C, H, W) * 255).astype(np.uint8)

    print("FVD on random data =", execute_default_fvd(real_frames, gen_frames))
