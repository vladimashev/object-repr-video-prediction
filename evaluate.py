from fvd import execute_default_fvd
import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from lpips import LPIPS


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

@torch.no_grad()
def evaluate_autoencoder(model, dataloader, device='cuda', lpips_net='alex'):
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    # LPIPS model (expects inputs in [-1,1])
    lpips_fn = LPIPS(net=lpips_net).to(device)
    lpips_fn.eval()

    total_mse = 0.0
    total_mae = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    total_lpips = 0.0

    total_frames = 0  # frames across all videos (B * T)
    total_batches = 0

    # For FVD
    real_sequences = []
    gen_sequences = []

    for batch in dataloader:
        # dataloader yields (imgs, masks) or imgs
        # (depending on dataset variant)
        isList = isinstance(batch, (tuple, list))
        if isList:
            imgs, masks = batch[0], batch[1]
            masks = masks.to(device)
        else:
            imgs = batch

        imgs = imgs.to(device)

        if isList:
            model_input = (imgs, masks)
        else:
            model_input = imgs

        recon = model(model_input)
        if isinstance(recon, (tuple, list)):
            recon = recon[0]
        recon = recon.to(device)

        # MSE/MAE loss over whole batch/time
        mse = F.mse_loss(recon, imgs)
        mae = F.l1_loss(recon, imgs)
        total_mse += float(mse.item())
        total_mae += float(mae.item())
        total_batches += 1
        
        B, T, C, H, W = imgs.shape

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
                psnr_val = psnr(x, y, data_range=255)

                total_ssim += float(ssim_val)
                total_psnr += float(psnr_val)
                total_frames += 1

        # collect sequences for FVD
        real_np = imgs_uint8.copy().astype(np.uint8) # зачем?
        gen_np = recon_uint8.copy().astype(np.uint8) # зачем?
        real_np = real_np.astype(np.float32)
        gen_np = gen_np.astype(np.float32)
        real_sequences.append(real_np)
        gen_sequences.append(gen_np)

    avg_mse = total_mse / total_batches
    avg_mae = total_mae / total_batches
    avg_ssim = total_ssim / max(1, total_frames)
    avg_psnr = total_psnr / max(1, total_frames)
    avg_lpips = total_lpips / max(1, total_frames)

    results = {
        "MSE": avg_mse,
        "MAE": avg_mae,
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

        results["FVD"] = float(fvd_value) if fvd_value is not None else -1

    return results

@torch.no_grad()
def evaluate_ar_transformer(model, dataloader, device, steps=15):
    """
    Оцениваем ТОЛЬКО кадры (frames).
    Ожидаемый batch: (frames, masks), где:
      frames: [B, T, C, H, W] float в [0,1]
      masks:  [B, T, H, W]     long/uint (НЕ используется в метриках)
    """
    model.eval()

    criterion = torch.nn.L1Loss()
    lpips_metric = LPIPS(net="alex").to(device)

    total_loss = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    total_lpips = 0.0
    count = 0

    real_sequences = []
    gen_sequences = []

    for _, batch in enumerate(dataloader):
        # ---- распаковка батча ----
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            frames, masks = batch
            frames = frames.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)
        else:
            # fallback: только кадры
            frames = batch.to(device, non_blocking=True)
            masks  = None

        # ---- разбиение на контекст и будущее ----
        context_f = frames[:, :5]              # [B, 5, C, H, W]
        future_f  = frames[:, 5:5+steps]       # [B, steps, C, H, W]
        context_m = masks[:,  :5] if masks is not None else None

        B = frames.size(0)
        preds_f = []

        # текущее окно контекста
        cur_context = (context_f.clone(), context_m.clone() if context_m is not None else None)

        # ---- авторегрессия ----
        for _ in range(steps):
            if cur_context[1] is not None:
                # модель возвращает (pred_frames, pred_masks)
                out_f, out_m, _ = model(cur_context)       # ([B,5,C,H,W], [B,5,H,W])
                next_f = out_f[:, -1]                   # [B,C,H,W]
                next_m = out_m[:, -1]                   # [B,H,W]
                # обновляем контекст (сдвиг окна по времени)
                cur_context = (
                    torch.cat([cur_context[0][:, 1:], next_f.unsqueeze(1)], dim=1),
                    torch.cat([cur_context[1][:, 1:], next_m.unsqueeze(1)], dim=1),
                )
            else:
                # если модель принимает только frames
                out_f = model(cur_context[0])            # [B,5,C,H,W]
                next_f = out_f[:, -1]                    # [B,C,H,W]
                cur_context = (torch.cat([cur_context[0][:, 1:], next_f.unsqueeze(1)], dim=1), None)

            preds_f.append(next_f.unsqueeze(1))          # [B,1,C,H,W]

        preds = torch.cat(preds_f, dim=1)                # [B, steps, C, H, W]

        # ---- метрики по кадрам ----
        loss = criterion(preds, future_f)
        total_loss += loss.item()

        imgs_uint8 = to_uint8_np(future_f)
        recon_uint8 = to_uint8_np(preds)
        
        for j in range(B):
            for t in range(steps):
                x = future_f[j, t].detach().cpu().numpy().transpose(1, 2, 0)  # [H,W,C]
                y = preds[j, t].detach().cpu().numpy().transpose(1, 2, 0)

                x_u8 = (np.clip(x, 0, 1) * 255).astype("uint8")
                y_u8 = (np.clip(y, 0, 1) * 255).astype("uint8")

                total_ssim  += ssim(x_u8, y_u8, data_range=255, channel_axis=-1)
                total_psnr  += psnr(x_u8, y_u8, data_range=255)

                # LPIPS ждёт тензоры в [-1,1]
                pred_norm = preds[j, t].unsqueeze(0).to(device) * 2 - 1  # [1,C,H,W]
                gt_norm   = future_f[j, t].unsqueeze(0).to(device) * 2 - 1
                total_lpips += float(lpips_metric(pred_norm, gt_norm).item())

                count += 1

        # for FVD
        real_np = imgs_uint8.astype(np.float32)
        gen_np = recon_uint8.astype(np.float32)
        real_sequences.append(real_np)
        gen_sequences.append(gen_np)
        
    # усреднение
    avg_loss  = total_loss / max(len(dataloader), 1)
    avg_ssim  = total_ssim / max(count, 1)
    avg_psnr  = total_psnr / max(count, 1)
    avg_lpips = total_lpips / max(count, 1)


    # real_sequences  [_, B, T, C, H, W]
    # [[video1], [video2]] -> [video1, video2]
    # [[gt1], [gt2]] ->       [gt1,     gt2]
    real_all = np.concatenate(real_sequences, axis=0)  # [B, T, C, H, W]
    gen_all = np.concatenate(gen_sequences, axis=0)

    # execute_default_fvd expects frames array shape [B, T, C, H, W], accepts torch or numpy.
    # pass device string
    try:
        fvd_value = execute_default_fvd(real_all, gen_all, device=str(device))
    except Exception as e:
        # If FVD computation fails, return None but don't crash
        fvd_value = None
        print(f"Warning: FVD computation failed: {e}")
        
    return {
        "Loss":  avg_loss,
        "SSIM":  avg_ssim,
        "PSNR":  avg_psnr,
        "LPIPS": avg_lpips,
        "FVD": float(fvd_value) if fvd_value is not None else -1
    }

# @torch.no_grad()
# def evaluate_ar_transformer(model, dataloader, device, steps=15):
#     model.eval()

#     criterion = torch.nn.L1Loss()
#     lpips_metric = LPIPS(net="alex").to(device)

#     total_loss = 0.0
#     total_ssim = 0.0
#     total_psnr = 0.0
#     total_lpips = 0.0
#     count = 0

#     preds_all, gts_all = [], []

#     for _, batch in enumerate(dataloader):
#         batch = batch.to(device)  # [B, T = 20, C, H, W]
#         context, future = batch[:, :5], batch[:, 5:]  # (B,5,..), (B,15,..)

#         B, _, C, H, W = context.shape
#         preds = []

#         # autoregressive rollout
#         cur_context = context.clone()
#         for t in range(steps):
#             out = model(cur_context)              # [B, T = 5, C, H, W]
#             next_frame = out[:, -1]               # last frame
#             preds.append(next_frame.unsqueeze(1)) # [B, 1, C, H, W]

#             # add prediction to the context
#             cur_context = torch.cat([cur_context[:, 1:], next_frame.unsqueeze(1)], dim=1)

#         preds = torch.cat(preds, dim=1)  # (B, 15, C, H, W)

#         # calculate metrics
#         loss = criterion(preds, future)
#         total_loss += loss.item()

#         for j in range(B):
#             for t in range(steps):
#                 x = future[j, t].detach().cpu().numpy().transpose(1, 2, 0)
#                 y = preds[j, t].detach().cpu().numpy().transpose(1, 2, 0)

#                 x_clipped = (np.clip(x, 0, 1) * 255).astype("uint8")
#                 y_clipped = (np.clip(y, 0, 1) * 255).astype("uint8")

#                 total_ssim += ssim(x_clipped, y_clipped, data_range=255, channel_axis=-1)
#                 total_psnr += psnr(x_clipped, y_clipped, data_range=255)

#                 # LPIPS
#                 pred_norm = preds[j, t].unsqueeze(0) * 2 - 1
#                 gt_norm = future[j, t].unsqueeze(0) * 2 - 1
#                 total_lpips += lpips_metric(pred_norm, gt_norm).item()

#                 count += 1

#     del preds, future, batch, context, cur_context, out, next_frame
#     torch.cuda.empty_cache()

#     #TODO: add fvd
    
#     avg_loss = total_loss / len(dataloader)
#     avg_ssim = total_ssim / count
#     avg_psnr = total_psnr / count
#     avg_lpips = total_lpips / count

#     return {
#         "Loss": avg_loss,
#         "SSIM": avg_ssim,
#         "PSNR": avg_psnr,
#         "LPIPS": avg_lpips,
#     }
