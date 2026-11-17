import numpy as np
import pandas as pd
import pytest
import torch

from core.anomaly import anomaly_scores, flag
from core.augment import make_view, random_mask
from core.evaluate import evaluate_embedding, linear_probe_auc
from core.models import SSLModel
from core.preprocess import Preprocessor
from core.train import TrainConfig, nt_xent, train_ssl


@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 4, size=(3, 12))
    y = rng.integers(0, 3, 400)
    X = centers[y] + rng.normal(0, 1, size=(400, 12))
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(12)])
    return df, y


def test_preprocessor_handles_nan_constant_and_inf(blobs):
    df, _ = blobs
    df = df.copy()
    df.iloc[0, 0] = np.nan
    df.iloc[1, 1] = np.inf
    df["const"] = 3.0
    p = Preprocessor()
    X = p.fit_transform(df)
    assert np.isfinite(X).all()
    assert "const" not in p.feature_names
    assert X.max() <= 5.0 and X.min() >= -5.0
    assert p.transform(df.head(3)).shape == (3, X.shape[1])


def test_mask_ratio_and_zeroing():
    x = torch.ones(2000, 20)
    masked, mask = random_mask(x, 0.3)
    assert abs(mask.mean().item() - 0.3) < 0.03
    assert (masked[mask.bool()] == 0).all()
    v, _ = make_view(x, 0.1, 0.0)
    assert v.shape == x.shape


def test_nt_xent_prefers_aligned_views():
    z = torch.randn(32, 16)
    aligned = nt_xent(z, z + 0.01 * torch.randn_like(z), 0.2)
    shuffled = nt_xent(z, z[torch.randperm(32)], 0.2)
    assert aligned < shuffled


def test_training_reduces_loss(blobs):
    df, _ = blobs
    X = Preprocessor().fit_transform(df)
    cfg = TrainConfig(epochs=12, latent_dim=8, hidden=64, batch_size=64)
    model, hist = train_ssl(X, cfg)
    assert isinstance(model, SSLModel)
    assert hist["total"][-1] < hist["total"][0]


def test_labels_never_reach_encoder(blobs):
    """Labels live in a separate frame; the encoder input width equals the feature count only."""
    df, y = blobs
    labels = pd.DataFrame({"y": y})
    X = Preprocessor().fit_transform(df)
    assert X.shape[1] == df.shape[1]
    model, _ = train_ssl(X, TrainConfig(epochs=1, latent_dim=4, hidden=32))
    assert model.encoder[0].in_features == df.shape[1]
    assert "y" not in df.columns and "y" in labels.columns


def test_embeddings_separate_known_clusters(blobs):
    df, y = blobs
    X = Preprocessor().fit_transform(df)
    model, _ = train_ssl(X, TrainConfig(epochs=25, latent_dim=8, hidden=64, batch_size=64))
    model.eval()
    with torch.no_grad():
        Z = model.embed(torch.from_numpy(X)).numpy()
    m = evaluate_embedding(Z, 3, pd.DataFrame({"c": y.astype(str)}), {"cat_label": "c"})
    assert m["nmi"] > 0.5


def test_linear_probe_and_anomaly_scores():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    Z = rng.normal(size=(300, 5)) + y[:, None] * 2
    assert linear_probe_auc(Z, y) > 0.85
    Z2 = np.vstack([rng.normal(size=(200, 4)), rng.normal(8, 1, size=(5, 4))])
    s = anomaly_scores(Z2)
    assert s[-5:].mean() > 0.9
    assert flag(s, 5).sum() >= 10
