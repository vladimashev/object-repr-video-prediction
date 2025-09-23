import torch
import torch.nn as nn
from models.patchifier import Patchifier
from models.positional_encoding import PositionalEncoding
from models.transformer_block import TransformerBlock, SpatialTemporalBlock


class VideoARTransformer(nn.Module):
    def __init__(self, encoder, decoder, H, W, patch_size, embed_dim, attn_dim, num_heads, mlp_size, num_tf_layers_ar):
        super().__init__()
        self.H, self.W = H, W
        self.patch_size = patch_size

        # --- Encoder ---
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        # --- Autoregressive Transformer ---
        self.ar_transformer = nn.ModuleList([
            TransformerBlock(
                token_dim=embed_dim,
                attn_dim=attn_dim,
                num_heads=num_heads,
                mlp_size=mlp_size
            )
            for _ in range(num_tf_layers_ar)
        ])
        self.proj = nn.Linear(embed_dim, embed_dim)

        # --- Decoder ---
        self.decoder = decoder
        for p in self.decoder.parameters():
            p.requires_grad = False

    @staticmethod
    def build_causal_mask(num_frames: int, num_patches: int, device) -> torch.Tensor:
        """
        Блочная causal mask: кадр t видит только <= t.
        Патчи внутри кадра общаются свободно.
        """
        # time mask (T, T)
        time_mask = torch.triu(torch.ones(num_frames, num_frames, dtype=torch.bool), diagonal=1)
        # расширяем на патчи
        mask = time_mask.repeat_interleave(num_patches, dim=0).repeat_interleave(num_patches, dim=1)
        return mask.to(device)

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            preds: (B, T, C, H, W)  # [ŷ₂..ŷ_{T+1}]
        """
        B, T, C, H, W = x.shape

        # (1) Encoder
        with torch.no_grad():
            feats = self.encoder(x)  # (B, T, Np, D)

        B, T, Np, D = feats.shape
        tokens = feats.view(B, T * Np, D)  # (B, T*Np, D)

        # (2) causal mask over frames
        attn_mask = self.build_causal_mask(T, Np, tokens.device)

        # (3) AR Transformer
        for blk in self.ar_transformer:
            tokens = blk(tokens, attn_mask=attn_mask)
        pred_feats = self.proj(tokens).view(B, T, Np, D)

        # (4) Decoder
        preds = self.decoder(pred_feats)  # (B, T, C, H, W)

        return preds


class VideoFrameTransformer(nn.Module):
    def __init__(self, encoder, decoder,
                 H, W, patch_size,
                 embed_dim, attn_dim, num_heads,
                 mlp_size, num_tf_layers_ar):
        super().__init__()

        # --- Encoder ---
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        # --- Autoregressive Transformer (spatial+temporal) ---
        self.ar_transformer = nn.ModuleList([
            SpatialTemporalBlock(
                token_dim=embed_dim,
                attn_dim=attn_dim,
                num_heads=num_heads,
                mlp_size=mlp_size,
                grid=(H // patch_size, W // patch_size),
                max_len=(64 // patch_size) * (64 // patch_size) * 15, # max 15 frames of size (64, 64)
                causal=True
            )
            for _ in range(num_tf_layers_ar)
        ])
        self.proj = nn.Linear(embed_dim, embed_dim)

        # --- Decoder ---
        self.decoder = decoder
        for p in self.decoder.parameters():
            p.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            preds: (B, T, C, H, W)

    
        """
        # Encoder
        tokens = self.encoder(x)  # (B, T, Np, D)

        B, T, Np, D = tokens.shape

        # Spatial+Temporal transformer blocks
        for blk in self.ar_transformer:
            tokens = blk(tokens)

        pred_feats = self.proj(tokens)

        # Decoder
        preds = self.decoder(pred_feats)  # (B, T, C, H, W)

        return preds