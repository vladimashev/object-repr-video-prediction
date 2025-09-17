import numpy as np
from scipy import linalg
from pytorch_fid.inception import InceptionV3

BATCH_SIZE = 64

def get_inception_model(device="cuda"):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device).eval()
    return model


def get_activations(images, model, device="cuda", batch_size=BATCH_SIZE):
    """
    Extract InceptionV3 features. Accepts tensor of shape [N, 3, H, W].

    Returns numpy array of [N, 2048].
    """
    n_images = images.size(0)
    activations = []

    for i in range(0, n_images, batch_size):
        batch = images[i:i+batch_size].to(device)
        pred = model(batch)[0]
        activations.append(pred.squeeze(3).squeeze(2).cpu().numpy())
    
    return np.concatenate(activations, axis=0)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if not np.isfinite(covmean).all():
        covmean = covmean.real
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return (diff @ diff) + np.trace(sigma1 + sigma2 - 2 * covmean)


def compute_fvd_via_fid(real_frames, gen_frames, device="cuda"):
    """
    Accepts tensors of shape [B, T, C, H, W].
    """
    B, T, C, H, W = real_frames.shape
    model = get_inception_model(device)

    real_frames = real_frames.reshape(B*T, C, H, W)
    gen_frames  = gen_frames.reshape(B*T, C, H, W)

    act_r = get_activations(real_frames, model, device=device)
    act_g = get_activations(gen_frames, model, device=device)

    mu_r, sigma_r = np.mean(act_r, axis=0), np.cov(act_r, rowvar=False)
    mu_g, sigma_g = np.mean(act_g, axis=0), np.cov(act_g, rowvar=False)

    fvd = calculate_frechet_distance(mu_r, sigma_r, mu_g, sigma_g)
    
    return float(fvd)
