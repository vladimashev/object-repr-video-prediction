import os
import torch
from datetime import datetime

SAVED_MODELS_ROOT = "checkpoints"
def get_save_root():
    if(not os.path.exists(SAVED_MODELS_ROOT)):
        os.makedirs(SAVED_MODELS_ROOT)
        
    return SAVED_MODELS_ROOT

def get_default_model_name():
    return datetime.now().strftime(f"checkpoint-%H-%M_%d-%m-%Y")


def save_model(model, optimizer, scheduler = None, stats = {},
               model_name = get_default_model_name(),
               save_path = get_save_root()):
    """ Saving model checkpoint """
    
    if(not os.path.exists(save_path)):
        os.makedirs(save_path)
    savepath = f"{save_path}/{model_name}.pth"

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'stats': stats
    }, savepath)

    print(f"Saved {model_name}'s state to {savepath}")
    
    return savepath


def load_model(model, optimizer, model_name, save_path = get_save_root()):
    """ Loading pretrained checkpoint """
    savepath = f"{save_path}/{model_name}.pth"

    checkpoint = torch.load(savepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if (optimizer):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    stats = checkpoint["stats"]
    
    return model, optimizer, stats
