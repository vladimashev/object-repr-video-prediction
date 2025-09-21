from fvd import execute_default_fvd
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from lpips import LPIPS


@torch.no_grad()
def evaluate_autoencoder(model, dataloader, device='cuda', compute_fvd=True, lpips_net='alex'):
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    mse_criterion = torch.nn.MSELoss()

    # LPIPS model (expects inputs in [-1,1])
    lpips_fn = LPIPS(net=lpips_net).to(device)
    lpips_fn.eval()

    total_loss = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    total_lpips = 0.0

    total_frames = 0  # frames across all videos (B * T)
    total_batches = 0

    # For FVD if requested
    real_sequences = []
    gen_sequences = []

    for batch in dataloader:
        # dataloader yields (imgs, masks)
        imgs, masks = batch[0], batch[1]

        imgs = imgs.to(device)
        masks = masks.to(device)

        recon = model((imgs, masks))
        recon = recon.to(device)

        # MSE loss over whole batch/time
        loss = mse_criterion(recon, imgs)
        total_loss += float(loss.item())
        total_batches += 1

        B, T, C, H, W = imgs.shape

        # skimage expects HxWxC in uint8
        def to_uint8_np(tensor):
            t = tensor.detach().cpu().float()
            if t.max() > 1.1:
                # assume already 0-255
                t = torch.clamp(t, 0.0, 255.0)
                arr = t.byte().numpy()
            else:
                t = torch.clamp(t, 0.0, 1.0)
                arr = (t * 255.0).byte().numpy()
            return arr  # dtype=uint8

        # For LPIPS, need [-1,1] float32 torch tensors
        def to_lpips_tensor(tensor):
            t = tensor.detach().to(device).float()
            if t.max() > 1.1:
                # assume 0-255
                t = t / 255.0
            # now in [0,1] -> scale to [-1,1]
            t = (t * 2.0) - 1.0
            # reshape to [B*T, 3, H, W]
            return t.view(-1, C, H, W)

        imgs_lpips = to_lpips_tensor(imgs)
        recon_lpips = to_lpips_tensor(recon)

        batch_frames = B * T
        chunk = 32
        lpips_vals = []
        for i in range(0, batch_frames, chunk):
            a = imgs_lpips[i:i+chunk]
            b = recon_lpips[i:i+chunk]
            with torch.no_grad():
                v = lpips_fn(a, b)  # returns [N,1,1,1] or [N,1]
                v = v.view(v.shape[0]).cpu().numpy()
            lpips_vals.append(v)
        lpips_vals = np.concatenate(lpips_vals, axis=0)
        total_lpips += float(lpips_vals.sum())

        imgs_uint8 = to_uint8_np(imgs)
        recon_uint8 = to_uint8_np(recon)

        # iterate frames
        for b in range(B):
            for t in range(T):
                x = imgs_uint8[b, t].transpose(1, 2, 0)    # H,W,C
                y = recon_uint8[b, t].transpose(1, 2, 0)

                ssim_val = ssim(x, y, data_range=255, channel_axis=-1)
                # except TypeError:
                #     # older skimage uses multichannel arg
                #     ssim_val = ssim(x, y, data_range=255, multichannel=True)
                psnr_val = psnr(x, y, data_range=255)

                total_ssim += float(ssim_val)
                total_psnr += float(psnr_val)
                total_frames += 1

        # collect sequences for FVD
        real_np = imgs_uint8.copy().astype(np.uint8)
        gen_np = recon_uint8.copy().astype(np.uint8)
        real_np = real_np.astype(np.float32)
        gen_np = gen_np.astype(np.float32)
        real_sequences.append(real_np)
        gen_sequences.append(gen_np)

    avg_loss = total_loss / total_batches
    avg_ssim = total_ssim / max(1, total_frames)
    avg_psnr = total_psnr / max(1, total_frames)
    avg_lpips = total_lpips / max(1, total_frames)

    results = {
        "Loss": avg_loss,
        "SSIM": avg_ssim,
        "PSNR": avg_psnr,
        "LPIPS": avg_lpips,
    }


    if len(real_sequences) == 0:
        results["FVD"] = -1
    else:
        real_all = np.concatenate(real_sequences, axis=0)  # [N_videos, T, C, H, W]
        gen_all = np.concatenate(gen_sequences, axis=0)

        # execute_default_fvd expects frames array shape [B, T, C, H, W], accepts torch or numpy.
        # pass device string
        try:
            fvd_value = execute_default_fvd(real_all, gen_all, device=str(device))
        except Exception as e:
            # If FVD computation fails, return None but don't crash
            fvd_value = None
            print(f"Warning: FVD computation failed: {e}")

        results["FVD"] = float(fvd_value) if fvd_value is not None else 1

    return results



@torch.no_grad()
def evaluate_ar_transformer(model, dataloader, device, steps=15):
    model.eval()

    criterion = torch.nn.L1Loss()
    lpips_metric = LPIPS(net="alex").to(device)

    total_loss = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    total_lpips = 0.0
    count = 0

    preds_all, gts_all = [], []

    for _, batch in enumerate(dataloader):
        batch = batch.to(device)  # [B, T = 20, C, H, W]
        context, future = batch[:, :5], batch[:, 5:]  # (B,5,..), (B,15,..)

        B, _, C, H, W = context.shape
        preds = []

        # autoregressive rollout
        cur_context = context.clone()
        for t in range(steps):
            out = model(cur_context)              # [B, T = 5, C, H, W]
            next_frame = out[:, -1]               # last frame
            preds.append(next_frame.unsqueeze(1)) # [B, 1, C, H, W]

            # add prediction to the context
            cur_context = torch.cat([cur_context[:, 1:], next_frame.unsqueeze(1)], dim=1)

        preds = torch.cat(preds, dim=1)  # (B, 15, C, H, W)

        # calculate metrics
        loss = criterion(preds, future)
        total_loss += loss.item()

        for j in range(B):
            for t in range(steps):
                x = future[j, t].detach().cpu().numpy().transpose(1, 2, 0)
                y = preds[j, t].detach().cpu().numpy().transpose(1, 2, 0)

                x_clipped = (np.clip(x, 0, 1) * 255).astype("uint8")
                y_clipped = (np.clip(y, 0, 1) * 255).astype("uint8")

                total_ssim += ssim(x_clipped, y_clipped, data_range=255, channel_axis=-1)
                total_psnr += psnr(x_clipped, y_clipped, data_range=255)

                # LPIPS
                pred_norm = preds[j, t].unsqueeze(0) * 2 - 1
                gt_norm = future[j, t].unsqueeze(0) * 2 - 1
                total_lpips += lpips_metric(pred_norm, gt_norm).item()

                count += 1

    del preds, future, batch, context, cur_context, out, next_frame
    torch.cuda.empty_cache()

    #TODO: add fvd
    
    avg_loss = total_loss / len(dataloader)
    avg_ssim = total_ssim / count
    avg_psnr = total_psnr / count
    avg_lpips = total_lpips / count

    return {
        "Loss": avg_loss,
        "SSIM": avg_ssim,
        "PSNR": avg_psnr,
        "LPIPS": avg_lpips,
    }
