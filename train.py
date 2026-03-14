import yaml
import inspect
import torch

from src.models.encdec import EncoderDecoder

import torch
import torch.nn as nn
import torch.nn.functional as F
class Discriminator(nn.Module):
    def __init__(self, width: int = 256, k_size: int = 3):
        super(TinyModel, self).__init__()
        padding = k_size // 2
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=k_size, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=k_size, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=k_size, padding=padding)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=k_size, padding=padding)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=k_size, padding=padding)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_out1 = (width - k_size + 1) // 2
        self.conv_out2 = (self.conv_out1 - k_size + 2 * padding + 1) // 2
        self.conv_out3 = (self.conv_out2 - k_size + 2 * padding + 1) // 2
        self.conv_out4 = (self.conv_out3 - k_size + 2 * padding + 1) // 2
        self.conv_out5 = (self.conv_out4 - k_size + 2 * padding + 1) // 2
        self.fc1 = nn.Linear(in_features=64 * self.conv_out5 ** 2, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=4)
        self.fc4 = nn.Linear(in_features=4, out_features=1)

        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool(x)
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)
        x = F.relu(self.conv5(x))
        x = self.maxpool(x)
        x = x.view(-1, 64 * self.conv_out5 ** 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
    

# Training code here


if __name__ == '__main__':
    frequency = 256
    time = 1234
    n_classes = 5

    x = torch.randn((8, frequency, time))
    c = torch.tensor([1,2,3])

    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    encdec = EncoderDecoder(
        frequency_dim=frequency, 
        n_classes=n_classes,
        **config["model"]
    )

    out = encdec(x, c)
    print(out["reconstruction"].shape)

    n_param_enc = sum(p.numel() for p in encdec.encoder.parameters() if p.requires_grad)
    print(f"Number of parameters in encoder: {n_param_enc}")
    n_param_dec = sum(p.numel() for p in encdec.decoder.parameters() if p.requires_grad)
    print(f"Number of parameters in decoder: {n_param_dec}")
    print(encdec._init_values)
