import torch
import torch.nn as nn

from .downsample import FullyConvolutionalEncoder
from .transformer import TransformerBlock
from .film import FiLMLayer

class Encoder(nn.Module):
    def __init__(
        self, 
        frequency_dim: int,
        in_channels: int = 1,
        out_channels: int = 32,
        fcn_layers: int = 1,
        fcn_dropout: float = 0.0,
        tf_blocks: int = 1,
        d_model: int = 64,
        d_state: int = 32,
        expand: int = 2,
        pos_dropout: float = 0.0,
        ff_dropout: float = 0.0
    ):
        super(Encoder, self).__init__()
        self.d_model = d_model
        
        # Convolution Downsampling
        self.downsampling = FullyConvolutionalEncoder(
            out_channels=out_channels,
            num_layers=fcn_layers,
            in_channels=in_channels,
            p_dropout=fcn_dropout
        )

        self.in_dim = self.downsampling.mid_channels * frequency_dim // 4
        self.freq_proj = nn.Linear(self.in_dim, d_model)

        # Transformer Block (Global/Local context and Dynamics)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model = d_model,
                d_state = d_state,
                expand = expand,
                pos_dropout = pos_dropout,
                ff_dropout = ff_dropout
            ) for _ in range(tf_blocks)
        ])

        # Class embeddings and Conditioning
        self.film = FiLMLayer(d_model, d_model)

    def forward(self, x, c_embeddings):
        """
        Encodes a spectrogram input into a latent distributions.

        Args:
            x (torch.Tensor): Input spectrogram or raw acoustic features.
                Shape: [Batch, Frequency, Time]

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor):
                - z: Sampled latent vector using the reparameterization trick.
                - mu: The mean of the latent distribution.
                - log_std: The log variance of the latent distribution.
        """
        # Convolutional Encoding
        x = self.downsampling(x) # [Batch, Channel, Frequency, Time]

        # Frequency projection
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2)   # [B, T, C, F]
        x = x.reshape(B, T, C * F)  # [B, T, D]

        x = self.freq_proj(x)

        # Transformer processing
        for block in self.transformer_blocks:
            x = self.film(x, c_embeddings)
            x = block(x)
        
        return x