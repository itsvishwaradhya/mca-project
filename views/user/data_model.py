"""Pick / upload a dataset, review data quality, choose or quick-train a model."""
import pandas as pd
import streamlit as st

from core import db
from core.config import UPLOAD_DIR
from core.pipeline import execute_run, run_compatible
from core.preprocess import numeric_features
from core.train import TrainConfig
from core.ui import cached_dataset, current_user, disclaimer, header

user = current_user()
header("Data & Model", "Choose the multivariate dataset and the representation model to analyse.")
disclaimer()

datasets = db.list_datasets()
tab_pick, tab_upload = st.tabs(["Choose dataset", "Upload CSV"])

with tab_upload:
    st.write("Upload any de-identified multivariate table (one row per patient/record). "
             "Optional label columns are kept **for evaluation only** and never used in training.")
    f = st.file_uploader("CSV file", type="csv")
    if f is not None:
        raw = pd.read_csv(f)
        c1, c2 = st.columns(2)
        id_col = c1.selectbox("ID column", ["(use row number)", *raw.columns])
        label_cols = c2.multiselect("Label columns (evaluation only)", [c for c in raw.columns if c != id_col])
        name = st.text_input("Dataset name", value=f.name.rsplit(".", 1)[0])
        if st.button("Save dataset", type="primary"):
            data = raw.copy()
            data.index = data[id_col] if id_col != "(use row number)" else range(len(data))
            data.index.name = "RecordID"
            labels = data[label_cols].copy() if label_cols else None
            drop = [c for c in [id_col, *label_cols] if c in data.columns]
            feats = numeric_features(data.drop(columns=drop))
            if feats.shape[1] < 3 or len(feats) < 50:
                st.error("Need at least 50 rows and 3 numeric feature columns.")
            else:
                meta = {}
                if labels is not None:
                    for c in label_cols:
                        nun = labels[c].nunique()
                        if nun == 2 and "binary_label" not in meta:
                            meta["binary_label"] = c
                        elif 2 < nun <= 20 and "cat_label" not in meta:
                            meta["cat_label"] = c
                stamp = db.now().replace(":", "").replace(" ", "_")
                fp = UPLOAD_DIR / f"{stamp}_{user['username']}_features.csv"
                lp = UPLOAD_DIR / f"{stamp}_{user['username']}_labels.csv"
                feats.to_csv(fp)
                if labels is not None:
                    labels.to_csv(lp)
                db.add_dataset(name, fp, lp if labels is not None else None, meta, len(feats),
                               feats.shape[1], user["username"])
                db.log(user["username"], "upload_dataset", name)
                st.success(f"Saved '{name}' ({feats.shape[0]} rows, {feats.shape[1]} features). "
                           "Select it in the other tab.")

with tab_pick:
    if not datasets:
        st.warning("No datasets yet. Ask an admin to register the demo dataset, or upload a CSV.")
        st.stop()
    options = {d["id"]: f"{d['name']}  ({d['n_rows']} rows x {d['n_features']} features)" for d in datasets}
    default = st.session_state.get("dataset_id", datasets[0]["id"])
    ids = list(options)
    ds_id = st.selectbox("Dataset", ids, format_func=options.get, index=ids.index(default) if default in ids else 0)
    if st.session_state.get("dataset_id") != ds_id:
        st.session_state["dataset_id"] = ds_id
        st.session_state.pop("run_id", None)
    feats, labels, meta = cached_dataset(ds_id)

    st.subheader("Data quality")
    miss = feats.isna().mean().mul(100)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(feats):,}")
    c2.metric("Features", feats.shape[1])
    c3.metric("Overall missing", f"{feats.isna().values.mean() * 100:.1f}%")
    c4.metric("Evaluation labels", "yes" if labels is not None else "none")
    with st.expander("Most-missing features"):
        st.dataframe(miss.sort_values(ascending=False).head(15).rename("missing_%").round(1))
    with st.expander("Preview"):
        st.dataframe(feats.head(20))

    st.subheader("Model")
    runs = db.list_runs("WHERE status='done' AND dataset_id=? AND (published=1 OR created_by=?)",
                        (ds_id, user["username"]))
    # published models trained on a different dataset may still fit if the columns match
    runs += [r for r in db.list_runs("WHERE status='done' AND published=1 AND dataset_id!=?", (ds_id,))
             if run_compatible(r, feats)]
    if runs:
        mopts = {r["id"]: f"{r['name']} ({'published' if r['published'] else 'yours'}, run {r['id']})" for r in runs}
        rids = list(mopts)
        cur = st.session_state.get("run_id")
        run_id = st.selectbox("Trained model", rids, format_func=mopts.get, index=rids.index(cur) if cur in rids else 0)
        st.session_state["run_id"] = run_id
        st.success("Model selected. Continue to Explore, Patient states or At-risk patients.")
    else:
        st.info("No compatible published model for this dataset yet. Quick-train one below.")

    with st.expander("Quick-train a model on this dataset", expanded=not runs):
        c1, c2, c3 = st.columns(3)
        epochs = c1.slider("Epochs", 5, 60, 25)
        latent = c2.select_slider("Latent size", [8, 16, 32, 64], value=32)
        k = c3.slider("Expected states (k)", 2, 10, 5)
        if st.button("Train", type="primary"):
            cfg = TrainConfig(epochs=epochs, latent_dim=latent, n_clusters=k)
            rid = db.create_run(f"Quick-train by {user['username']}", ds_id, cfg.to_dict(), user["username"])
            db.log(user["username"], "quick_train", f"run {rid}")
            bar = st.progress(0.0, text="Training SSL encoder, then baselines...")
            execute_run(rid, progress_cb=lambda p: bar.progress(p, text=f"{p * 100:.0f}%"))
            st.session_state["run_id"] = rid
            st.rerun()
