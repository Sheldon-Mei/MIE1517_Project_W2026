import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionModule(nn.Module):
    """
    Convolutional module used in Conformer blocks:
    - LayerNorm + GLU
    - Pointwise Conv (expansion)
    - Depthwise Conv + BatchNorm + Activation
    - Pointwise Conv (projection)
    - Dropout

    It captures local and global temporal dependencies via:
        - GroupNorm (channel-wise normalization)
        - Pointwise Conv + GLU (channel expansion)
        - Depthwise Conv + BatchNorm + Swish (local temporal filtering)
        - Pointwise Conv projection
        - Dropout for regularization

    This module processes input of shape [B, C, T] where:
        - B = batch size
        - C = number of channels / features
        - T = sequence length (time steps)
    """

    def __init__(
        self, 
        d_input: int, 
        expansion_factor: int = 2, 
        kernel_size: int = 31, 
        dropout: float = 0.1
    ):
        """
        Parameters
        ----------
        d_input : int
            Number of input channels / features.
        expansion_factor : int, default=2
            Factor to expand channels in pointwise convolution before GLU.
        kernel_size : int, default=31
            Kernel size of the depthwise convolution.
        dropout : float, default=0.1
            Dropout probability applied after the final pointwise projection.
        """
        super(ConvolutionModule, self).__init__()

        self.group_norm = nn.GroupNorm(1, d_input)
        self.pointwise_conv1 = nn.Conv1d(d_input, d_input * expansion_factor * 2, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            d_input * expansion_factor, d_input * expansion_factor,
            kernel_size=kernel_size,
            groups=d_input * expansion_factor,  # depthwise
            padding=kernel_size // 2
        )
        self.batch_norm = nn.BatchNorm1d(d_input * expansion_factor)
        self.pointwise_conv2 = nn.Conv1d(d_input * expansion_factor, d_input, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, C, T] → [batch, channels, time]
        """
        # LayerNorm over channels
        x = self.group_norm(x)

        # Pointwise conv + GLU
        x = self.pointwise_conv1(x)          # [B, 2*C*expansion, T]
        x = F.glu(x, dim=1)                  # [B, C*expansion, T]

        # Depthwise conv + BN + Swish
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.silu(x)                        # Swish activation

        # Pointwise conv (projection back to original dim)
        x = self.pointwise_conv2(x)

        # Dropout
        x = self.dropout(x)

        return x