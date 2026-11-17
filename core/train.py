"""Self-supervised training: NT-Xent contrastive loss + masked-feature reconstruction."""
import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .augment import make_view
from .models import SSLModel


@dataclass
class TrainConfig:
    latent_dim: int = 32
    hidden: int = 256
    epochs: int = 60
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    noise_std: float = 0.1
    mask_ratio: float = 0.3
    temperature: float = 0.2
    recon_weight: float = 1.0  # lambda: weight of reconstruction vs contrastive loss
    n_clusters: int = 5
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """SimCLR loss: the two views of a row are positives, every other row in the batch is a negative."""
    n = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.T / temperature
    sim.fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])
    return F.cross_entropy(sim, targets)


def train_ssl(X: np.ndarray, cfg: TrainConfig, progress: Callable[[int, dict], None] | None = None):
    """Train on features only. Returns (model, history) with per-epoch total/contrastive/recon loss."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = SSLModel(X.shape[1], cfg.latent_dim, cfg.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    data = torch.from_numpy(X)
    hist = {"total": [], "contrastive": [], "recon": []}
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(data))
        sums = np.zeros(3); n = 0
        for i in range(0, len(data), cfg.batch_size):
            xb = data[perm[i:i + cfg.batch_size]]
            if len(xb) < 4:
                continue
            v1, m1 = make_view(xb, cfg.noise_std, cfg.mask_ratio)
            v2, _ = make_view(xb, cfg.noise_std, cfg.mask_ratio)
            _, p1, rec1 = model(v1)
            _, p2, _ = model(v2)
            contrastive = nt_xent(p1, p2, cfg.temperature)
            denom = m1.sum().clamp(min=1.0)
            recon = (((rec1 - xb) ** 2) * m1).sum() / denom  # loss only on hidden entries
            loss = contrastive + cfg.recon_weight * recon
            opt.zero_grad(); loss.backward(); opt.step()
            sums += np.array([loss.item(), contrastive.item(), recon.item()]) * len(xb); n += len(xb)
        row = sums / max(n, 1)
        for key, v in zip(("total", "contrastive", "recon"), row):
            hist[key].append(float(v))
        if progress:
            progress(epoch, {k: v[-1] for k, v in hist.items()})
    return model, hist


@torch.no_grad()
def ssl_outputs(model: SSLModel, X: np.ndarray):
    """Embeddings plus per-row reconstruction error on the *unmasked* input."""
    model.eval()
    x = torch.from_numpy(X)
    z, _, rec = model(x)
    return z.numpy(), ((rec - x) ** 2).mean(dim=1).numpy()


def main() -> None:
    from .pipeline import execute_run  # local import: avoids a cycle and torch cost on import

    ap = argparse.ArgumentParser(description="Execute a queued training run from the database.")
    ap.add_argument("--run-id", type=int, required=True)
    execute_run(ap.parse_args().run_id)


if __name__ == "__main__":
    main()
