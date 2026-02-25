import torch
import torch.nn as nn
import torch.nn.functional as F

# Uncompression r = 4
# ie. If latent dim = 128, then uncompresses to 512

class DepthRefinementBlock(nn.Module):
    def __init__(self, channels: int, p_dropout: float):
        super(DepthRefinementBlock, self).__init__()

        self.local_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3,1), padding=(1,0), groups=channels),
            nn.Conv2d(channels, channels, kernel_size=(1,15), padding=(0,7), groups=channels),
            nn.SiLU(),
            nn.Dropout2d(p_dropout),

            nn.Conv2d(channels, channels, kernel_size=(15,1), padding=(7,0), groups=channels),
            nn.Conv2d(channels, channels, kernel_size=(1,3), padding=(0,1), groups=channels),
            nn.SiLU(),
            nn.Dropout2d(p_dropout),
        )

        # Pointwise/Depthwise
        self.quality_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3, 31), padding=(1, 15), groups=channels),
            nn.Conv2d(channels, channels, kernel_size=(31, 3), padding=(15, 1), groups=channels),
            nn.SiLU(),
            nn.Dropout2d(p_dropout),

            nn.Conv2d(channels, channels * 4, kernel_size=1),
            nn.SiLU(),
            nn.Dropout2d(p_dropout),
            
            nn.Conv2d(channels * 4, channels, kernel_size=1),
            nn.SiLU(),
            nn.Dropout2d(p_dropout),
        )

    def forward(self, x):
        # Local Dynamics (Temporal + Freq)
        x = x + self.local_refine(x)
        # Texture Refinement
        x = x + self.quality_refine(x)
        return x

class ResolutionStage(nn.Module):
    """
    Increases Frequency and Time dimensions jointly.
    Contains N Depth Blocks for quality.
    """
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        num_layers: int,
        p_dropout: int
    ):
        super(ResolutionStage, self).__init__()
        self.mid_channels = int(out_channels * 0.75)
        self.start_channels = max(4, self.mid_channels // 2)

        # ---- alternating upsample / refinement ----
        upsample_layers = []

        # reverse channel flow for decoder
        ch_schedule = [self.mid_channels, self.start_channels]

        in_ch = in_channels

        for i, out_ch in enumerate(ch_schedule):

            # ---- Upsample ----
            if i % 2 == 0:
                # long frequency kernel
                upsample_layers.append(
                    nn.ConvTranspose2d(
                        in_ch,
                        out_ch,
                        kernel_size=(31, 3),
                        stride=(2, 2),
                        padding=(15, 1),
                        output_padding=(1, 1),
                    )
                )
            else:
                # long time kernel
                upsample_layers.append(
                    nn.ConvTranspose2d(
                        in_ch,
                        out_ch,
                        kernel_size=(3, 31),
                        stride=(2, 2),
                        padding=(1, 15),
                        output_padding=(1, 1),
                    )
                )

            upsample_layers.append(nn.BatchNorm2d(out_ch))
            upsample_layers.append(nn.SiLU())

            # ---- Refinement stack ----
            for _ in range(num_layers):
                upsample_layers.append(
                    DepthRefinementBlock(out_ch, p_dropout)
                )

            in_ch = out_ch

        self.upsample_layers = nn.Sequential(*upsample_layers)

    def forward(self, x):
        """
        Processes latent features to reconstruct the original vocalization signal.

        Args:
            x: Latent tensor [Batch, H, W]
            spatial_shape: Tuple (H, W) of the latent map before flattening

        Returns:
            torch.Tensor: Reconstruction (spectrogram-like).
                Shape: (Batch, Frequency, Time)
        """
        return self.upsample_layers(x)