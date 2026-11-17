"""Orchestration: load data, train SSL + baselines, evaluate, persist artifacts, and run inference."""
import json
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from . import db
from .anomaly import anomaly_scores
from .baselines import ae_outputs, fit_pca, train_autoencoder
from .config import MODELS_DIR
from .evaluate import evaluate_embedding
from .models import SSLModel
from .preprocess import Preprocessor
from .train import TrainConfig, ssl_outputs, train_ssl


def load_dataset(dataset_id: int) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    ds = db.get_dataset(dataset_id)
    if ds is None:
        raise ValueError(f"dataset {dataset_id} not found")
    feats = pd.read_csv(ds["features_path"], index_col=0)
    labels = pd.read_csv(ds["labels_path"], index_col=0).loc[feats.index] if ds["labels_path"] else None
    return feats, labels, ds["meta"]


def execute_run(run_id: int, progress_cb=None) -> None:
    """Train SSL + PCA + autoencoder on the run's dataset, evaluate, save artifacts, update the DB row."""
    run = db.get_run(run_id)
    cfg = TrainConfig.from_dict(run["config"])
    t0 = time.time()
    try:
        db.update_run(run_id, status="running", progress=0.0, error=None)
        feats, labels, meta = load_dataset(run["dataset_id"])
        prep = Preprocessor()
        X = prep.fit_transform(feats)  # labels never touch X

        total_epochs = cfg.epochs * 2  # SSL + autoencoder phases
        history: dict = {"ssl": {}, "ae": []}

        def report(done: int):
            db.update_run(run_id, progress=min(done / total_epochs, 0.99), history=history)
            if progress_cb:
                progress_cb(done / total_epochs)

        def ssl_cb(epoch, losses):
            for k, v in losses.items():
                history["ssl"].setdefault(k, []).append(v)
            if epoch % 2 == 0 or epoch == cfg.epochs - 1:
                report(epoch + 1)

        model, _ = train_ssl(X, cfg, ssl_cb)

        def ae_cb(epoch, loss):
            history["ae"].append(loss)
            if epoch % 2 == 0 or epoch == cfg.epochs - 1:
                report(cfg.epochs + epoch + 1)

        ae, _ = train_autoencoder(X, cfg, ae_cb)

        z_ssl, err_ssl = ssl_outputs(model, X)
        z_ae, err_ae = ae_outputs(ae, X)
        _, z_pca, err_pca = fit_pca(X, cfg.latent_dim, cfg.seed)

        k = cfg.n_clusters
        metrics = {"k": k, "n_rows": int(len(X)), "n_features": int(X.shape[1]), "seconds": None}
        for name, Z, err in (("ssl", z_ssl, err_ssl), ("pca", z_pca, err_pca), ("ae", z_ae, err_ae)):
            metrics[name] = evaluate_embedding(Z, k, labels, meta, anomaly_scores(Z, err, cfg.seed), cfg.seed)
        metrics["raw"] = evaluate_embedding(X, k, labels, meta, anomaly_scores(X, None, cfg.seed), cfg.seed)
        metrics["seconds"] = round(time.time() - t0, 1)

        out = MODELS_DIR / f"run_{run_id}"
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"state": model.state_dict(), "n_features": X.shape[1], "cfg": cfg.to_dict()}, out / "ssl_model.pt")
        joblib.dump(prep, out / "preprocessor.joblib")
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        db.update_run(run_id, status="done", progress=1.0, history=history, metrics=metrics,
                      model_dir=str(out), finished_at=db.now())
    except Exception:
        db.update_run(run_id, status="failed", error=traceback.format_exc()[-2000:], finished_at=db.now())
        raise


def run_compatible(run: dict, feats: pd.DataFrame) -> bool:
    """True if the run's saved preprocessor can be applied to this dataset's columns."""
    try:
        prep = joblib.load(Path(run["model_dir"]) / "preprocessor.joblib")
        return prep.compatible(feats)
    except Exception:
        return False


class LoadedModel:
    """A trained SSL model + its preprocessor, ready for inference on any compatible dataframe."""

    def __init__(self, run_id: int):
        run = db.get_run(run_id)
        if not run or run["status"] != "done":
            raise ValueError(f"run {run_id} is not finished")
        d = Path(run["model_dir"])
        ckpt = torch.load(d / "ssl_model.pt", map_location="cpu", weights_only=False)
        cfg = TrainConfig.from_dict(ckpt["cfg"])
        self.model = SSLModel(ckpt["n_features"], cfg.latent_dim, cfg.hidden)
        self.model.load_state_dict(ckpt["state"])
        self.model.eval()
        self.prep: Preprocessor = joblib.load(d / "preprocessor.joblib")
        self.cfg = cfg

    def compatible(self, df: pd.DataFrame) -> bool:
        return self.prep.compatible(df)

    def infer(self, df: pd.DataFrame, seed: int = 42):
        """Returns scaled features, embeddings, anomaly scores in [0,1]."""
        X = self.prep.transform(df)
        Z, err = ssl_outputs(self.model, X)
        return X, Z, anomaly_scores(Z, err, seed)
