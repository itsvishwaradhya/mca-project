"""Compare finished runs and methods side by side."""
import pandas as pd
import plotly.express as px
import streamlit as st

from core import db
from core.ui import header, require_admin

require_admin()
header("Experiments", "SSL vs PCA vs Autoencoder across runs.")

runs = db.list_runs("WHERE status='done'")
if not runs:
    st.info("No finished runs yet. Launch one on the Training runs page.")
    st.stop()

NAMES = {"ssl": "SSL", "pca": "PCA", "ae": "Autoencoder", "raw": "Raw"}
METRICS = ["silhouette", "davies_bouldin", "nmi", "ari", "probe_auc", "anomaly_auc"]

chosen = st.multiselect("Runs", runs, default=runs[:3], format_func=lambda r: f"{r['id']} - {r['name']}")
rows = []
for r in chosen:
    for key, label in NAMES.items():
        if key in r["metrics"]:
            rows.append({"run": f"{r['id']} {r['name']}", "method": label, **r["metrics"][key],
                         "latent": r["config"].get("latent_dim"), "epochs": r["config"].get("epochs"),
                         "mask": r["config"].get("mask_ratio"), "lambda": r["config"].get("recon_weight")})
if not rows:
    st.stop()
df = pd.DataFrame(rows)
st.dataframe(df.round(3), hide_index=True, use_container_width=True)

metric = st.selectbox("Metric to chart", [m for m in METRICS if m in df.columns], index=min(4, len(METRICS) - 1))
st.plotly_chart(px.bar(df, x="run", y=metric, color="method", barmode="group", height=420), use_container_width=True)
st.caption("Davies-Bouldin: lower is better. All other metrics: higher is better.")

best = df[df.method == "SSL"].sort_values("probe_auc", ascending=False).head(1) if "probe_auc" in df else None
if best is not None and len(best):
    st.success(f"Best SSL run by linear-probe AUROC: **{best.iloc[0]['run']}** ({best.iloc[0]['probe_auc']:.3f}).")
st.download_button("Download table (CSV)", df.to_csv(index=False), "experiments.csv", "text/csv")
