import torch
import torch.nn as nn

# Compression rate = 4

class ProcessingBlock(nn.Module):
    def __init__(self, channels, p_dropout=0.0):
        super(ProcessingBlock, self).__init__()

        self.block = nn.Sequential(
            # Depthwise
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(17, 17),
                padding=(8, 8),
                groups=channels
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(),

            nn.Dropout(p=p_dropout),

            # Pointwise
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)

class FullyConvolutionalEncoder(nn.Module):
    def __init__(
        self, 
        out_channels=32,
        num_layers=2,
        in_channels=1, 
        p_dropout=0.0
    ):
        super(FullyConvolutionalEncoder, self).__init__()
        self.mid_channels = int(out_channels * 0.75)
        self.start_channels = max(4, self.mid_channels // 2)

        # ---- alternating downsample / processing ----
        downsample_layers = []

        in_ch = in_channels
        ch_schedule = [self.start_channels, self.mid_channels]

        for i, out_ch in enumerate(ch_schedule):

            # ---- Downsample ----
            # alternate time vs frequency emphasis
            if i % 2 == 0:
                # long frequency kernel
                downsample_layers.append(
                    nn.Conv2d(
                        in_ch,
                        out_ch,
                        kernel_size=(31, 3),
                        stride=(2, 2),
                        padding=(15, 1),
                    )
                )
            else:
                # long time kernel
                downsample_layers.append(
                    nn.Conv2d(
                        in_ch,
                        out_ch,
                        kernel_size=(3, 31),
                        stride=(2, 2),
                        padding=(1, 15),
                    )
                )

            downsample_layers.append(nn.BatchNorm2d(out_ch))
            downsample_layers.append(nn.SiLU())

            # ---- Processing stack ----
            for _ in range(num_layers):
                downsample_layers.append(ProcessingBlock(out_ch, p_dropout))

            in_ch = out_ch

        self.downsample_layers = nn.Sequential(*downsample_layers)

    def forward(self, x):
        # input shape: [Batch, Frequency, Time]
        
        x = x.unsqueeze(1) # [Batch, Channel, Frequency, Time]

        x = self.downsample_layers(x)

        return x