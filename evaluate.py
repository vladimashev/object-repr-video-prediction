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
