import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # Represents the distance between our latent dist and a Standard Normal N(0,1)
    p = Normal(mu, torch.exp(0.5 * logvar))
    q = Normal(torch.zeros_like(mu), torch.ones_like(logvar))
    kld_loss = kl_divergence(p, q).sum()
    
    return (recon_loss + kld_loss) / x.size(0)