import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encdec import EncoderDecoder


# Training code here


if __name__ == '__main__':
    frequency = 256
    time = 1234
    n_classes = 5

    x = torch.randn((8, frequency, time))
    c = torch.tensor([2])

    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    encdec = EncoderDecoder(
        frequency_dim=frequency, 
        n_classes=n_classes,
        **config["model"]
    )

    encdec.eval()
    with torch.no_grad():
        out = encdec(x, c)
    print(out["reconstruction"].shape)

    n_param = sum(p.numel() for p in encdec.parameters())
    print(f"Number of parameters in model: {n_param}")