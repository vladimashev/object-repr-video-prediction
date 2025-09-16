import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

@torch.no_grad()
def evaluate_autoencoder(model, dataloader, device):
    model.eval()

    total_loss = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    count = 0

    criterion = torch.nn.MSELoss()

    for _, batch in enumerate(dataloader):
        batch = batch.to(device)
        recon = model(batch)

        loss = criterion(recon, batch)
        total_loss += loss.item()

        for j in range(batch.size(0)):
            x = batch[j].cpu().numpy().transpose(1, 2, 0)
            y = recon[j].cpu().numpy().transpose(1, 2, 0)

            x_clipped = (x * 255).astype("uint8")
            y_clipped = (y * 255).astype("uint8")

            total_ssim += ssim(x_clipped, y_clipped, data_range=255, channel_axis=-1)
            total_psnr += psnr(x_clipped, y_clipped, data_range=255)

            count += 1

    avg_loss = total_loss / len(dataloader)
    avg_ssim = total_ssim / count
    avg_psnr = total_psnr / count

    return {
        "Loss": avg_loss,
        "SSIM": avg_ssim,
        "PSNR": avg_psnr,
    }

@torch.no_grad()
def evaluate_ar_transformer(model, dataloader, device, steps=15):
    model.eval()

    criterion = torch.nn.MSELoss()
    lpips_metric = LPIPS(net="alex").to(device)

    total_loss = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    total_lpips = 0.0
    count = 0

    preds_all, gts_all = [], []

    for _, batch in enumerate(dataloader):
        batch = batch.to(device)  # (B, 20, C, H, W)
        context, future = batch[:, :5], batch[:, 5:]  # (B,5,..), (B,15,..)

        B, _, C, H, W = context.shape
        preds = []

        # autoregressive rollout
        cur_context = context.clone()
        for t in range(steps):
            out = model(cur_context)              # (B, Tp, C, H, W), Tp=5
            next_frame = out[:, -1]               # последний кадр
            preds.append(next_frame.unsqueeze(1)) # (B,1,C,H,W)

            # сдвигаем контекст
            cur_context = torch.cat([cur_context[:, 1:], next_frame.unsqueeze(1)], dim=1)

        preds = torch.cat(preds, dim=1)  # (B, 15, C, H, W)

        # === Metrics ===
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

                # LPIPS (переводим в [-1,1])
                pred_norm = preds[j, t].unsqueeze(0) * 2 - 1
                gt_norm = future[j, t].unsqueeze(0) * 2 - 1
                total_lpips += lpips_metric(pred_norm, gt_norm).item()

                count += 1

        preds_all.append(preds.cpu())
        gts_all.append(future.cpu())

    # === FVD ===
    preds_all = torch.cat(preds_all, dim=0)  # (N,15,C,H,W)
    gts_all = torch.cat(gts_all, dim=0)

    # pytorch-fvd ожидает (N, T, C, H, W) в [0,1]
    pred_feats = get_fvd_feats(preds_all.to(device), device, use_inception_v3=True)
    gt_feats = get_fvd_feats(gts_all.to(device), device, use_inception_v3=True)
    fvd_score = fvd(pred_feats, gt_feats)

    avg_loss = total_loss / len(dataloader)
    avg_ssim = total_ssim / count
    avg_psnr = total_psnr / count
    avg_lpips = total_lpips / count

    return {
        "Loss": avg_loss,
        "SSIM": avg_ssim,
        "PSNR": avg_psnr,
        "LPIPS": avg_lpips,
        "FVD": fvd_score.item()
    }
