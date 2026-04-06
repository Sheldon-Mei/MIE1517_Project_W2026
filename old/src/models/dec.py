import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .upsample import ResolutionStage
from .transformer import TransformerBlock
from .film import FiLMLayer
    
class Decoder(nn.Module):
    def __init__(
        self, 
        frequency_dim: int = 256,
        out_channels: int = 32,
        num_layers: int = 3,
        fcn_dropout: float = 0.0,
        d_model: int = 128,
        d_state: int = 32,
        expand: int = 2,
        pos_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        tf_blocks: int = 2,
        d_latent: int = 32
    ):
        super(Decoder, self).__init__()

        self.resolution = ResolutionStage(1, out_channels, num_layers, fcn_dropout)

        scaled_freq = d_latent * 4
        in_dim = self.resolution.start_channels * scaled_freq
        self.feature_proj = nn.Linear(in_dim, frequency_dim)
        self.film = FiLMLayer(d_model, frequency_dim)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                frequency_dim, 
                d_state, 
                expand, 
                pos_dropout, 
                ff_dropout
            ) for _ in range(tf_blocks)
        ])

    def forward(self, z, cond):
        """
        Decodes a latent vector back into spectrogram.

        Args:
            z (torch.Tensor): Sampled latent vector [Batch, Seq_Len, d_latent]
            c_embeddings (torch.Tensor): Conditioning embeddings.

        Returns:
            torch.Tensor: Reconstructed spectrogram [Batch, Frequency, Time]
        """
        z = z.unsqueeze(1) # [B, C, T, F]
        B, C, T, F = z.shape
        z = z.view(B, C, F, T) # [B, C, F, T]

        # Iterate through Resolution Stages
        x = self.resolution(z)

        B, C, F, T = x.shape
        x = x.view(B, T, C * F)
        x = self.feature_proj(x)

        # Iterate through Transformer Stages (for processing time)
        for block in self.transformer_blocks:
            x = self.film(x, cond)
            x = block(x)

        B, T, F = x.shape
        x = x.view(B, F, T)
        return x