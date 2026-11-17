"""PCA and plain autoencoder baselines."""
from collections.abc import Callable

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn

from .models import AutoEncoder


def fit_pca(X: np.ndarray, n_components: int, seed: int = 42):
    pca = PCA(n_components=min(n_components, X.shape[1]), random_state=seed).fit(X)
    Z = pca.transform(X)
    recon_err = ((X - pca.inverse_transform(Z)) ** 2).mean(axis=1)
    return pca, Z, recon_err


def train_autoencoder(X: np.ndarray, cfg, progress: Callable[[int, float], None] | None = None):
    torch.manual_seed(cfg.seed)
    model = AutoEncoder(X.shape[1], cfg.latent_dim, cfg.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    data = torch.from_numpy(X)
    losses = []
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(data))
        tot, n = 0.0, 0
        for i in range(0, len(data), cfg.batch_size):
            xb = data[perm[i:i + cfg.batch_size]]
            if len(xb) < 2:
                continue
            _, rec = model(xb)
            loss = nn.functional.mse_loss(rec, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(xb); n += len(xb)
        losses.append(tot / max(n, 1))
        if progress:
            progress(epoch, losses[-1])
    return model, losses


@torch.no_grad()
def ae_outputs(model: AutoEncoder, X: np.ndarray):
    model.eval()
    x = torch.from_numpy(X)
    z, rec = model(x)
    return z.numpy(), ((rec - x) ** 2).mean(dim=1).numpy()
