import torch
import torch.nn as nn
import torch.nn.functional as F

from .enc import Encoder
from .latent_head import LatentHead
from .dec import Decoder

class EncoderDecoder(nn.Module):
    """
    A unified Transformer-based Encoder-Decoder architecture for signal processing.

    This model integrates a conditional embedding mechanism with a symmetric 
    encoder-decoder structure. It is designed to map input signals into a 
    low-dimensional latent manifold and reconstruct them, guided by class-specific 
    contextual information.

    Attributes:
        name (str): The identifier for the model architecture.
        c_embeddings (nn.Embedding): Lookup table for class-conditioned vectors.
        encoder (Encoder): The feature extraction and compression module.
        decoder (Decoder): The reconstruction and upsampling module.
    """

    def __init__(
        self, 
        frequency_dim: int = 256,
        n_classes: int = 5,
        out_channels: int = 32,
        num_processing_blocks_enc: int = 2,
        fcn_dropout_enc: float = 0.0,
        num_processing_blocks_dec: int = 2,
        fcn_dropout_dec: float = 0.0,
        tf_blocks_enc: int = 2,
        tf_blocks_dec: int = 2,
        d_model: int = 64,
        d_state: int = 32,
        expand: int = 2,
        pos_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        d_latent: float = 32
    ):
        """
        Initializes the EncoderDecoder with specified structural hyperparameters.

        Args:
            frequency_dim: The spatial or spectral resolution of the input signal.
            n_classes: Number of distinct categories for conditional embedding.
            in_channels: Number of input feature maps (e.g., 1 for mono audio).
            out_channels: Intermediate filter depth for convolutional stages.
            fcn_layers_enc/dec: Number of convolutional processing blocks.
            fcn_dropout_enc/dec: Spatial dropout probability for conv layers.
            tf_blocks_enc/dec: Number of Transformer/State-Space blocks.
            d_model: Internal hidden dimension for the attention mechanism.
            d_state: State dimension for SSM-based layers.
            n_heads: Number of parallel attention heads.
            expand: Expansion factor for the feed-forward or SSM networks.
            pos_dropout: Dropout applied to positional encodings.
            attn_dropout: Dropout applied to the attention weight matrix.
            ff_dropout: Dropout applied to the pointwise feed-forward network.
            d_latent: The bottleneck dimension of the latent representation.
        """
        super(EncoderDecoder, self).__init__()

        self.c_embeddings = nn.Embedding(
            num_embeddings=n_classes, 
            embedding_dim=d_model
        )

        self.encoder = Encoder(
            frequency_dim=frequency_dim,
            out_channels=out_channels,
            fcn_layers=num_processing_blocks_enc,
            fcn_dropout=fcn_dropout_enc,
            tf_blocks=tf_blocks_enc,
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            pos_dropout=pos_dropout,
            ff_dropout=ff_dropout
        )

        # Gaussian Latent distribution
        self.latent_head = LatentHead(d_model, d_latent)

        self.decoder = Decoder(
            frequency_dim=frequency_dim,
            out_channels=out_channels,
            num_layers=num_processing_blocks_dec,
            fcn_dropout=fcn_dropout_dec,
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            tf_blocks=tf_blocks_dec,
            pos_dropout=pos_dropout,
            ff_dropout=ff_dropout,
            d_latent=d_latent
        )

    def forward(self, x, c):
        """
        Executes the conditional Variational Autoencoder (cVAE).

        Args:
            x (Tensor): Input signal of shape (Batch, Freq, Time).
            c (Tensor): Class indices for conditioning of shape (Batch,).

        Returns:
            dict: A collection of output tensors containing:
                - "reconstruction": The generated signal output.
                - "latent": The sampled latent vector (z).
                - "mu": The mean of the latent distribution.
                - "std": The standard deviation of the latent distribution.
        """
        c = self.c_embeddings(c)
        x = self.encoder(x, c)
        z, mu, std = self.latent_head(x) # [Batch, Time, Frequency]
        y = self.decoder(z, c)
        return {
            "reconstruction": y,
            "latent": z,
            "mu": mu,
            "std": std,
        }