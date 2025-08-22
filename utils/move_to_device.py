import torch

def move_to_device(data, device):
    """Safely move any nested structure of tensors to device"""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (tuple, list)):
        return type(data)(move_to_device(x, device) for x in data)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    else:
        return data
    