"""Shared Streamlit helpers: auth guard, active dataset/model context, cached computations."""
import numpy as np
import pandas as pd
import streamlit as st

from . import db
from .config import DISCLAIMER
from .evaluate import best_k, cluster_profile, kmeans_labels
from .pipeline import LoadedModel, load_dataset


def current_user() -> dict:
    u = st.session_state.get("user")
    if not u:
        st.stop()
    return u


def require_admin() -> dict:
    u = current_user()
    if u["role"] != "admin":
        st.error("Admin access required.")
        st.stop()
    return u


def header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def disclaimer() -> None:
    st.caption(f":material/info: {DISCLAIMER}")


@st.cache_data(show_spinner=False)
def cached_dataset(dataset_id: int):
    return load_dataset(dataset_id)


@st.cache_resource(show_spinner=False)
def cached_model(run_id: int) -> LoadedModel:
    return LoadedModel(run_id)


@st.cache_data(show_spinner="Computing representations...")
def cached_inference(run_id: int, dataset_id: int):
    feats, _, _ = load_dataset(dataset_id)
    return cached_model(run_id).infer(feats)


@st.cache_data(show_spinner="Projecting to 2D (UMAP)...")
def cached_projection(run_id: int, dataset_id: int) -> np.ndarray:
    _, Z, _ = cached_inference(run_id, dataset_id)
    try:
        import umap
        return umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(Z)
    except Exception:  # umap unavailable -> PCA fallback
        from sklearn.decomposition import PCA
        return PCA(2, random_state=42).fit_transform(Z)


@st.cache_data(show_spinner=False)
def cached_clusters(run_id: int, dataset_id: int, k: int) -> np.ndarray:
    _, Z, _ = cached_inference(run_id, dataset_id)
    return kmeans_labels(Z, k)


@st.cache_data(show_spinner="Choosing number of states...")
def cached_best_k(run_id: int, dataset_id: int):
    _, Z, _ = cached_inference(run_id, dataset_id)
    return best_k(Z)


def context(need_model: bool = True):
    """Return the active (dataset_id, run_id) or stop the page with guidance."""
    current_user()
    ds_id, run_id = st.session_state.get("dataset_id"), st.session_state.get("run_id")
    if ds_id is None or (need_model and run_id is None):
        st.info("Pick a dataset and a model first on the **Data & Model** page.")
        st.stop()
    run = db.get_run(run_id) if run_id else None
    ds = db.get_dataset(ds_id)
    st.sidebar.caption(f"Dataset: **{ds['name']}**" + (f"  \nModel: **{run['name']}**" if run else ""))
    return ds_id, run_id


def profile_for(run_id: int, dataset_id: int, k: int) -> pd.DataFrame:
    X, _, _ = cached_inference(run_id, dataset_id)
    _, labels, meta = cached_dataset(dataset_id)
    names = cached_model(run_id).prep.feature_names
    return cluster_profile(X, names, cached_clusters(run_id, dataset_id, k), labels, meta)
