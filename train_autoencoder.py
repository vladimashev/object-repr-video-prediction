import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from save_load import save_model

from evaluate_autoencoder import evaluate_autoencoder

# from torchvision.utils import save_image
# import os

def train_epoch(model, train_loader, optimizer, criterion, device, mode):
    """ Training a model for one epoch """
    
    loss_list = []
    for batch in train_loader:
        img_batch = batch[mode]
        img_batch = img_batch.to(device)
        recon = model(img_batch)
        loss = criterion(recon, img_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        
    mean_loss = np.mean(loss_list)
    
    return mean_loss, loss_list


def train_autoencoder(model, train_loader, val_loader, device,
                      epochs=30, lr=1e-3, mode="img",
                      logger=None, log_image=None, save_name=False,
                      start_epoch=0):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    train_loss = []
    val_loss =  []
    loss_iters = []
    valid_acc = []

    for epoch in tqdm(range(epochs)):
        model.train()
        mean_loss, cur_loss_iters = train_epoch(model, train_loader,
                                                optimizer, criterion,
                                                device, mode)
        train_loss.append(mean_loss)

        if logger:
            logger(f'Loss/Train', mean_loss, global_step=epoch+start_epoch)

        loss_iters = loss_iters + cur_loss_iters
        
        if ((epoch+start_epoch+1) % 5 == 0 or epoch==epochs-1):
            print(f"\n Epoch {epoch+1}/{epochs}")
            evaluate_autoencoder(model, val_loader, device, "img",
                                 epoch+start_epoch, logger, log_image, save_name)

            if save_name:
                save_model(model, optimizer, { "epoch": epoch+start_epoch }, save_name)
            
        scheduler.step(mean_loss)
    
    print(f"Training completed")
    
    return train_loss, val_loss, loss_iters, valid_acc
