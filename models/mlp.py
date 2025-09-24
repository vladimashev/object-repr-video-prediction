import torch.nn as nn

class MLP(nn.Module):
    """
    2-Layer Multi-Layer Perceptron used in transformer blocks
    
    Args:
    -----
    in_dim: int
        Dimensionality of the input embeddings to the MLP
    hidden_dim: int
        Hidden dimensionality of the MLP
    """
    
    def __init__(self, in_dim, hidden_dim):
        """ MLP Initializer """
        super().__init__()
        self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, in_dim),
            )
        
    def forward(self, x):
        """ Forward """
        y = self.mlp(x)
        return y


# class MLP(nn.Module):
#     """
#     2-Layer Multi-Layer Perceptron used in transformer blocks
    
#     Args:
#     -----
#     in_dim: int
#         Dimensionality of the input embeddings to the MLP
#     hidden_dim: int
#         Hidden dimensionality of the MLP
#     """
    
#     def __init__(self, in_dim, hidden_dim, drop=0.3):
#         """ MLP Initializer """
#         super().__init__()
#         self.mlp = nn.Sequential(
#                 nn.Linear(in_dim, hidden_dim),
#                 nn.GELU(),
#                 nn.Dropout(drop),
#                 nn.Linear(hidden_dim, in_dim),
#                 nn.Dropout(drop)

#             )
        
#     def forward(self, x):
#         """ Forward """
#         y = self.mlp(x)
#         return y