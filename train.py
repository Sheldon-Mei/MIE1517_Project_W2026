import yaml
import inspect
import torch

from src.models.encdec import EncoderDecoder


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