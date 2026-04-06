import torch
import torch.nn as nn

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation Layer. 
    
    Affine transformation of features 'x' based on a condition 'z'.
    FiLM(x) = γ(z) * x + β(z)
    """
    def __init__(
        self, 
        input_dim: int,
        d_model: int
    ):
        super(FiLMLayer, self).__init__()
        self.gamma = nn.Linear(input_dim, d_model)
        self.beta = nn.Linear(input_dim, d_model)

    def forward(self, x, z):
        """
        x: [batch, seq_len, feature_dim]
        z: [batch] (categorical class indices)
        """
        return self.gamma(z) * x + self.beta(z)