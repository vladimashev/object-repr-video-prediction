import torch
import torch.nn as nn
from models.patchifier import Patchifier
from models.positional_encoding import PositionalEncoding


class ViT(nn.Module):
    """ 
    Vision Transformer for image classification
    """

    def __init__(self, patch_size, max_len, token_dim, attn_dim, num_heads, mlp_size, num_tf_layers, num_classes):
        """ Model initializer """
        super().__init__()

        # breaking image into patches, and projection to transformer token dimension
        self.pathchifier = Patchifier(patch_size)
        self.patch_projection = nn.Sequential(
                nn.LayerNorm(patch_size * patch_size * 3),
                nn.Linear(patch_size * patch_size * 3, token_dim)
            )

        # adding CLS token and positional embedding
        self.cls_token = nn.Parameter(torch.randn(1, token_dim) / (token_dim ** 0.5), requires_grad=True)
        self.pos_emb = PositionalEncoding(token_dim, max_len)

        # cascade of transformer blocks
        transformer_blocks = [
            TransformerBlock(
                    token_dim=token_dim,
                    attn_dim=attn_dim,
                    num_heads=num_heads,
                    mlp_size=mlp_size
                )
            for _ in range(num_tf_layers)
        ]
        self.transformer_blocks = nn.Sequential(*transformer_blocks)

        # classifier
        self.classifier = nn.Linear(token_dim, num_classes)
        return

    
    def forward(self, x):
        """ 
        Forward pass
        """
        B = x.shape[0]  # (B, T, C, H, W)
        
        # breaking image into patches, and projection to transformer token dimension
        patches = self.pathchifier(x)  # (B, T*num_patches, patch_dim)
        patch_tokens = self.patch_projection(patches)  # (B, 16, D) ????

        # concatenating CLS token and adding positional embeddings
        cur_cls_token = self.cls_token.unsqueeze(0).repeat(B, 1, 1)
        tokens = torch.cat([cur_cls_token, patch_tokens], dim=1)  # ~(B, 1 + 16, D)
        tokens_with_pe = self.pos_emb(tokens)

        # processing with transformer
        out_tokens = self.transformer_blocks(tokens_with_pe)
        out_cls_token = out_tokens[:, 0]  # fetching only CLS token

        # classification
        logits = self.classifier(out_cls_token)
        return logits


    def get_attn_mask(self):
        """
        Fetching the last attention maps from all TF Blocks
        """
        attn_masks = [tf.get_attention_masks() for tf in self.transformer_blocks]
        return attn_masks