import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba import MambaModule
from .conv_module import ConvolutionModule
from .positional_encoding import PositionalEncoding

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        pos_dropout: float = 0.1,
        ff_dropout: float = 0.1,
    ):
        super(TransformerBlock, self).__init__()

        self.name = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, pos_dropout)
        
        # Mamba Branch
        self.norm1 = nn.RMSNorm(d_model)
        self.mamba = MambaModule(
            d_model=d_model, 
            d_state=d_state
        )

        # Convolution Module
        self.conv_module = ConvolutionModule(d_model)

        # Feed-Forward Branch
        self.norm3 = nn.RMSNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * expand),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(d_model * expand, d_model)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        return

    def forward(
        self, 
        x: torch.Tensor, 
        residual_global: torch.Tensor=None,
    ):
        """
        Forward pass of the Mamba-Transformer block.
        
        Processes input through a sequential chain of Mamba (local temporal dynamics), 
        Attention (global context), and Feed-Forward (feature refinement) layers 
        with residual connections.

        Args:
            x (torch.Tensor): Input sequence of shape [Batch, Frequency, Time].
                Typically represents the current sequence being processed or generated.
            h_prev (torch.Tensor, optional): The hidden state from the previous 
                SSM step of shape [Batch, d_state, d_inner]. Required for 
                recurrent/inference mode. Defaults to None.
            residual_global (torch.Tensor, optional): External acoustic
                features of shape [Batch, Frequency, Time]. If provided,
                the Attention layer operates in 'Cross-Attention' mode, using 
                this tensor as Key and Value to anchor the generation. 
                Defaults to None (Self-Attention mode).

        Returns:
            tuple(torch.Tensor, torch.Tensor):
                - out: Processed sequence of shape [Batch, Time_Local, d_model].
                - h_next: Updated SSM hidden state for the next time step or 
                  sequence summary.
        """
        # --- Mamba (Temporal Dynamics) ---
        residual = x
        x_norm = self.norm1(x)
        x_mamba = self.mamba(x_norm)
        x = x_mamba + residual

        # --- Convolution Module ---
        residual = x
        x = x.permute(0, 2, 1)
        x = self.conv_module(x)
        x = x.permute(0, 2, 1)
        x = x + residual

        # --- FF ---
        residual = x
        x_norm = self.norm3(x)
        x_ff = self.ff(x_norm)
        x = x_ff + residual

        return x