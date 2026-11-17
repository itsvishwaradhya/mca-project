"""Configure, launch and monitor SSL training runs; publish models for analysts."""
import subprocess
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

from core import db
from core.config import ROOT
from core.train import TrainConfig
from core.ui import header, require_admin

admin = require_admin()
header("Training runs", "Set the self-supervised hyperparameters, launch a run in the background and watch it train.")

datasets = db.list_datasets()
if not datasets:
    st.warning("Register a dataset first.")
    st.stop()


def launch(run_id: int) -> None:
    log_dir = ROOT / "artifacts" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    with open(log_dir / f"run_{run_id}.log", "w") as f:
        subprocess.Popen([sys.executable, "-m", "core.train", "--run-id", str(run_id)], cwd=ROOT,
                         stdout=f, stderr=subprocess.STDOUT, creationflags=flags)


with st.expander("New run", expanded=True):
    d = TrainConfig()
    with st.form("new_run"):
        c1, c2 = st.columns(2)
        ds = c1.selectbox("Dataset", datasets, format_func=lambda x: f"{x['id']} - {x['name']}")
        name = c2.text_input("Run name", value="SSL baseline")
        c1, c2, c3, c4 = st.columns(4)
        latent = c1.select_slider("Latent dim", [8, 16, 32, 64, 128], value=d.latent_dim)
        epochs = c2.slider("Epochs", 5, 200, d.epochs)
        batch = c3.select_slider("Batch size", [64, 128, 256, 512], value=d.batch_size)
        lr = c4.select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3], value=d.lr)
        c1, c2, c3, c4 = st.columns(4)
        noise = c1.slider("Noise std", 0.0, 0.5, d.noise_std, 0.05)
        mask = c2.slider("Mask ratio", 0.0, 0.8, d.mask_ratio, 0.05)
        temp = c3.slider("Temperature", 0.05, 1.0, d.temperature, 0.05)
        lam = c4.slider("Recon weight (lambda)", 0.0, 5.0, d.recon_weight, 0.1)
        k = st.slider("Clusters (k) for evaluation", 2, 10, d.n_clusters)
        if st.form_submit_button("Launch run", type="primary"):
            cfg = TrainConfig(latent_dim=latent, epochs=epochs, batch_size=batch, lr=lr, noise_std=noise,
                              mask_ratio=mask, temperature=temp, recon_weight=lam, n_clusters=k)
            rid = db.create_run(name, ds["id"], cfg.to_dict(), admin["username"])
            launch(rid)
            db.log(admin["username"], "launch_run", f"run {rid}: {name}")
            st.success(f"Run {rid} started.")


@st.fragment(run_every=3)
def monitor() -> None:
    runs = db.list_runs()
    if not runs:
        st.caption("No runs yet.")
        return
    tbl = pd.DataFrame([{"id": r["id"], "name": r["name"], "status": r["status"],
                         "progress_%": round(r["progress"] * 100), "published": bool(r["published"]),
                         "epochs": r["config"].get("epochs"), "latent": r["config"].get("latent_dim"),
                         "by": r["created_by"], "created": r["created_at"]} for r in runs])
    st.dataframe(tbl, hide_index=True, use_container_width=True)
    sel = st.selectbox("Inspect run", runs, format_func=lambda r: f"{r['id']} - {r['name']} ({r['status']})")
    st.progress(float(sel["progress"]))
    h = sel["history"]
    if h.get("ssl"):
        loss = pd.DataFrame({"epoch": range(1, len(h["ssl"]["total"]) + 1), **{f"SSL {k}": v for k, v in h["ssl"].items()}})
        st.plotly_chart(px.line(loss.melt("epoch"), x="epoch", y="value", color="variable", height=300),
                        use_container_width=True)
    if sel["status"] == "failed" and sel["error"]:
        st.error("Run failed")
        st.code(sel["error"])
    c1, c2, _ = st.columns([2, 1, 2])
    if sel["status"] == "done":
        if sel["published"]:
            if c1.button("Unpublish", key=f"unpub{sel['id']}"):
                db.update_run(sel["id"], published=0)
                db.log(admin["username"], "unpublish", f"run {sel['id']}")
                st.rerun()
        elif c1.button("Publish to analysts", type="primary", key=f"pub{sel['id']}"):
            db.update_run(sel["id"], published=1)
            db.log(admin["username"], "publish", f"run {sel['id']}")
            st.rerun()
    if sel["status"] != "running" and c2.button("Delete run", key=f"del{sel['id']}"):
        db.ex("DELETE FROM runs WHERE id=?", (sel["id"],))
        db.log(admin["username"], "delete_run", f"run {sel['id']}")
        st.rerun()


st.subheader("Runs")
monitor()
