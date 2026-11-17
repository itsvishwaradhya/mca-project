"""Augmentation and masking strategies for tabular (multivariate) rows."""
import torch


def gaussian_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    return x + std * torch.randn_like(x)


def random_mask(x: torch.Tensor, ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero out a random subset of features. Returns (masked_x, mask) where mask=1 marks hidden entries."""
    mask = (torch.rand_like(x) < ratio).float()
    return x * (1 - mask), mask


def make_view(x: torch.Tensor, noise_std: float, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    """One stochastic view = gaussian noise + feature masking."""
    masked, mask = random_mask(x, mask_ratio)
    return gaussian_noise(masked, noise_std), mask
