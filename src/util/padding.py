import torch

def pad_class_indices(
    x: torch.Tensor, 
    class_length: int
):
    """
    x: tensor like [1] or [1,2]
    returns: [class_length]
    """
    x = x.long()

    if x.numel() >= class_length:
        return x[:class_length]

    padded = torch.zeros(class_length, dtype=torch.long, device=x.device)
    padded[:x.numel()] = x
    return padded

def pad_conv(kernel_size):
    """
    Computes same padding for Conv2d.
    """
    if isinstance(kernel_size, int):
        return kernel_size // 2

    return tuple(k // 2 for k in kernel_size)