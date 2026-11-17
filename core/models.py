"""MLP encoder / decoder / projection head."""
import torch
from torch import nn


def mlp(sizes: list[int], last_act: bool = False) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers += [nn.BatchNorm1d(sizes[i + 1]), nn.ReLU()]
    return nn.Sequential(*layers)


class SSLModel(nn.Module):
    def __init__(self, n_features: int, latent_dim: int = 32, hidden: int = 256, proj_dim: int = 32):
        super().__init__()
        self.n_features, self.latent_dim, self.hidden = n_features, latent_dim, hidden
        self.encoder = mlp([n_features, hidden, hidden // 2, latent_dim])
        self.projector = mlp([latent_dim, latent_dim, proj_dim])  # used only by the contrastive loss
        self.decoder = mlp([latent_dim, hidden // 2, hidden, n_features])

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return z, self.projector(z), self.decoder(z)


class AutoEncoder(nn.Module):
    """Plain autoencoder baseline: same encoder/decoder shape, no augmentation, no contrastive loss."""

    def __init__(self, n_features: int, latent_dim: int = 32, hidden: int = 256):
        super().__init__()
        self.n_features, self.latent_dim, self.hidden = n_features, latent_dim, hidden
        self.encoder = mlp([n_features, hidden, hidden // 2, latent_dim])
        self.decoder = mlp([latent_dim, hidden // 2, hidden, n_features])

    def embed(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)
