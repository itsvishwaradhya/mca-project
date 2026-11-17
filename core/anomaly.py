"""Anomaly / at-risk scoring on learned representations."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import IsolationForest


def _rank01(a: np.ndarray) -> np.ndarray:
    return (rankdata(a) - 1) / max(len(a) - 1, 1)


def anomaly_scores(Z: np.ndarray, recon_err: np.ndarray | None = None, seed: int = 42) -> np.ndarray:
    """Score in [0,1] (higher = more unusual): rank-average of
    IsolationForest on the embedding and the reconstruction error (if available)."""
    iso = IsolationForest(n_estimators=200, random_state=seed, n_jobs=1).fit(Z)
    s = _rank01(-iso.score_samples(Z))
    if recon_err is not None:
        s = 0.5 * s + 0.5 * _rank01(recon_err)
    return s


def flag(scores: np.ndarray, percentile: float) -> np.ndarray:
    """Boolean mask of the top `percentile`% most anomalous rows."""
    thr = np.percentile(scores, 100 - percentile)
    return scores >= thr


def top_deviations(Xs: np.ndarray, names: list[str], i: int, k: int = 5) -> pd.DataFrame:
    """Which standardised features of row i deviate most (z-scores)."""
    z = Xs[i]
    idx = np.argsort(-np.abs(z))[:k]
    return pd.DataFrame({"feature": [names[j] for j in idx], "z_score": z[idx].round(2)})
