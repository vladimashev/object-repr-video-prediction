import math
import torch.nn as nn
import torch

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional encoding 

    Args:
    -----
    d_model: int
        Dimensionality of the slots/tokens
    max_len: int
        Length of the sequence
    """

    def __init__(self, d_model, max_len):
        """
        Initializing the positional encoding
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # initializing embedding
        self.pe = self._get_pe()
        return

    def _get_pe(self):
        """
        Initializing the temporal positional encoding given the encoding mode
        """
        max_len = self.max_len
        d_model = self.d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(1, max_len, d_model)
        return pe

    def forward(self, x):
        """
        Adding the positional encoding to the input tokens of the transformer
        """
        if x.device != self.pe.device:
            self.pe = self.pe.to(x.device)
        batch_size, num_tokens = x.shape[0], x.shape[1]
        cur_pe = self.pe.repeat(batch_size, 1, 1)[:, :num_tokens]

        y = x + cur_pe
        return y


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, grid_size):
        """
        Args:
            d_model: int — размер эмбеддинга
            grid_size: tuple (H_p, W_p) — сетка патчей по высоте и ширине
        """
        super().__init__()
        self.d_model = d_model
        self.H_p, self.W_p = grid_size

        assert d_model % 4 == 0, "d_model должно делиться на 4 для 2D синусоиды"

        self.pe = self._build_2d_pe(self.H_p, self.W_p, d_model)  # (Np, D)

    def _build_2d_pe(self, H_p, W_p, d_model):
        """
        Строим 2D синусоидальную PE для сетки H_p x W_p
        """
        pe = torch.zeros(H_p, W_p, d_model)

        # координаты
        y_pos = torch.arange(0, H_p, dtype=torch.float).unsqueeze(1)  # (H_p,1)
        x_pos = torch.arange(0, W_p, dtype=torch.float).unsqueeze(1)  # (W_p,1)

        div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() *
                             (-math.log(10000.0) / (d_model // 2)))

        # вертикальная компонента (y)
        pe[:, :, 0::4] = torch.sin(y_pos * div_term).unsqueeze(1)   # (H_p,1,D/2)
        pe[:, :, 1::4] = torch.cos(y_pos * div_term).unsqueeze(1)

        # горизонтальная компонента (x)
        pe[:, :, 2::4] = torch.sin(x_pos * div_term).unsqueeze(0)   # (1,W_p,D/2)
        pe[:, :, 3::4] = torch.cos(x_pos * div_term).unsqueeze(0)

        pe = pe.view(H_p * W_p, d_model)  # (Np,D)
        return pe.unsqueeze(0).unsqueeze(0)  # (1,1,Np,D)

    def forward(self, x):
        """
        Args:
            x: (B, T, Np, D)
        Returns:
            y: (B, T, Np, D) с добавленной spatial PE
        """
        B, T, Np, D = x.shape
        device = x.device
        pe = self.pe.to(device)  # (1,1,Np,D)
        pe = pe.expand(B, T, Np, D)  # дублируем для батча и кадров
        return x + pe