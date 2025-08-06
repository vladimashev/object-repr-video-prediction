import os
import torch
import torchvision
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

@torch.no_grad()
def evaluate_autoencoder(model, dataloader, device, mode,
                         epoch = None, logger=None, log_image=None, model_name=''):
    should_save = log_image and epoch is not None

    if should_save:
        os.makedirs("imgs", exist_ok=True)
        os.makedirs("imgs/training", exist_ok=True)
        os.makedirs(f"imgs/training/{model_name}", exist_ok=True)
    
    model.eval()

    total_loss = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    count = 0

    criterion = torch.nn.MSELoss()

    for _, batch in enumerate(dataloader):
        batch = batch[mode].to(device)
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

    print(f"Validation Loss:  {avg_loss:.4f}")
    print(f"Avg SSIM:         {avg_ssim:.4f}")
    print(f"Avg PSNR:         {avg_psnr:.2f} dB")
    
    if logger:
        logger(f'Loss/Valid', avg_loss, global_step=epoch)
        logger(f'SSIM/Valid', avg_ssim, global_step=epoch)
        logger(f'PSNR/Valid', avg_psnr, global_step=epoch)

    if should_save:
        grid = torchvision.utils.make_grid(model(batch))
        log_image('images', grid, global_step=epoch)
        torchvision.utils.save_image(grid, os.path.join(os.getcwd(), "imgs", "training", model_name, f"imgs_{epoch+1}.png"))

    return {
        "loss": avg_loss,
        "ssim": avg_ssim,
        "psnr": avg_psnr,
    }
