"""Representation-quality metrics. Labels are used here only, never during training."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score, normalized_mutual_info_score,
                             roc_auc_score, silhouette_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def kmeans_labels(Z: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)


def best_k(Z: np.ndarray, k_range=range(2, 11), seed: int = 42) -> tuple[int, dict[int, float]]:
    """Pick k by silhouette on a subsample (fast)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), size=min(len(Z), 2000), replace=False)
    scores = {}
    for k in k_range:
        lab = kmeans_labels(Z[idx], k, seed)
        scores[k] = float(silhouette_score(Z[idx], lab))
    return max(scores, key=scores.get), scores


def linear_probe_auc(Z: np.ndarray, y: np.ndarray, seed: int = 42) -> float:
    """5-fold cross-validated AUROC of a logistic regression on frozen embeddings."""
    y = np.asarray(y)
    mask = ~pd.isna(y)
    Zm, ym = Z[mask], y[mask].astype(int)
    if len(np.unique(ym)) < 2 or np.bincount(ym).min() < 5:
        return float("nan")
    p = cross_val_predict(LogisticRegression(max_iter=2000), Zm, ym,
                          cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    return float(roc_auc_score(ym, p))


def anomaly_auc(scores: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y)
    mask = ~pd.isna(y)
    if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
        return float("nan")
    return float(roc_auc_score(y[mask].astype(int), scores[mask]))


def evaluate_embedding(Z: np.ndarray, k: int, labels: pd.DataFrame | None, meta: dict,
                       scores: np.ndarray | None = None, seed: int = 42) -> dict:
    cl = kmeans_labels(Z, k, seed)
    m = {"silhouette": float(silhouette_score(Z, cl, sample_size=min(len(Z), 3000), random_state=seed)),
         "davies_bouldin": float(davies_bouldin_score(Z, cl))}
    if labels is not None:
        cat, binary = meta.get("cat_label"), meta.get("binary_label")
        if cat and cat in labels:
            valid = labels[cat].notna().values
            m["nmi"] = float(normalized_mutual_info_score(labels[cat][valid], cl[valid]))
            m["ari"] = float(adjusted_rand_score(labels[cat][valid], cl[valid]))
        if binary and binary in labels:
            m["probe_auc"] = linear_probe_auc(Z, labels[binary].values, seed)
            if scores is not None:
                m["anomaly_auc"] = anomaly_auc(scores, labels[binary].values)
    return m


def cluster_profile(feat_scaled: np.ndarray, names: list[str], clusters: np.ndarray,
                    labels: pd.DataFrame | None, meta: dict, top: int = 3) -> pd.DataFrame:
    """One row per discovered state: size, outcome rates and the features that define it."""
    mu = feat_scaled.mean(axis=0)
    rows = []
    for c in sorted(np.unique(clusters)):
        sel = clusters == c
        diff = feat_scaled[sel].mean(axis=0) - mu
        order = np.argsort(-np.abs(diff))[:top]
        desc = ", ".join(f"{'high' if diff[j] > 0 else 'low'} {names[j]}" for j in order)
        row = {"state": int(c), "patients": int(sel.sum()), "share_%": round(100 * sel.mean(), 1), "defining_features": desc}
        if labels is not None:
            b = meta.get("binary_label")
            if b and b in labels:
                row[f"{b}_rate_%"] = round(100 * float(pd.to_numeric(labels[b][sel], errors="coerce").mean()), 1)
            for s in meta.get("score_cols", []):
                if s in labels:
                    row[f"mean_{s}"] = round(float(pd.to_numeric(labels[s][sel], errors="coerce").replace(-1, np.nan).mean()), 1)
        rows.append(row)
    return pd.DataFrame(rows)
