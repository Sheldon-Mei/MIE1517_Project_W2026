import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssm import StateSpaceModel

class MambaModule(nn.Module):
    """
    Description:
        For time-feature representation learning.
        
    Brief:
        [Input] Temporal features [N, A, T, F]
        [Output] Temporal features [N, A, T, F]

    Args:
        d_model (int): Dimension of the hidden layer
        d_state (int):
    """
    def __init__(
        self, 
        d_model: int, 
        d_state: int
    ):
        super(MambaModule, self).__init__()

        # Computation
        self.in_proj = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Local-feature convolution
        self.conv1d = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_model, 
            kernel_size=3, 
            padding=1,
            groups=d_model
        )

        self.silu = nn.SiLU()

        # Communication
        self.ssm = StateSpaceModel(d_model, d_state)

    # def forward(self, u: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u shape: [B, C, d_model]

        # --- Main Branch ---
        x = self.in_proj(u)

        x = x.transpose(1, 2)  # [B, d_model, Time]
        x = self.conv1d(x)
        x = self.silu(x)
        x = x.transpose(1, 2)  # Back to [B, Time, d_model]

        # x_ssm, h_next = self.ssm(h_prev, x)
        x_ssm = self.ssm(x)

        # --- Gating Branch ---
        z = self.gate_proj(u)
        z = self.silu(z)

        # --- Output ---
        y = self.out_proj(x_ssm * z)
        
        # return y, h_next
        return y