"""SSL vs PCA vs plain autoencoder (vs raw features) on the model's training dataset."""
import pandas as pd
import plotly.express as px
import streamlit as st

from core import db
from core.ui import context, header

header("Compare methods", "Is the self-supervised representation actually better than the baselines?")
_, run_id = context()
run = db.get_run(run_id)
m = run["metrics"]
if not m:
    st.warning("This run has no metrics yet.")
    st.stop()

LABELS = {"ssl": "Self-supervised (ours)", "pca": "PCA", "ae": "Autoencoder", "raw": "Raw features"}
METRICS = {
    "silhouette": ("Silhouette", True), "davies_bouldin": ("Davies-Bouldin", False),
    "nmi": ("NMI vs ICU type", True), "ari": ("ARI vs ICU type", True),
    "probe_auc": ("Linear-probe AUROC", True), "anomaly_auc": ("Anomaly AUROC", True),
}
rows = [{"method": LABELS[k], **{METRICS[x][0]: m[k].get(x) for x in METRICS if x in m[k]}} for k in LABELS if k in m]
df = pd.DataFrame(rows).set_index("method")
st.dataframe(df.round(3), use_container_width=True)
st.caption(f"{m['n_rows']:,} rows, {m['n_features']} features, k = {m['k']}. "
           "Higher is better except Davies-Bouldin (lower is better). Labels were used only to compute NMI/ARI/AUROC.")

long = df.reset_index().melt(id_vars="method", var_name="metric", value_name="value").dropna()
fig = px.bar(long, x="metric", y="value", color="method", barmode="group", height=420)
st.plotly_chart(fig, use_container_width=True)

ssl = df.loc[LABELS["ssl"]]
wins = [c for c in df.columns if c != "Davies-Bouldin" and ssl[c] >= df.loc[[LABELS["pca"], LABELS["ae"]], c].max()]
if "Davies-Bouldin" in df.columns and ssl["Davies-Bouldin"] <= df.loc[[LABELS["pca"], LABELS["ae"]], "Davies-Bouldin"].min():
    wins.append("Davies-Bouldin")
st.info(f"The self-supervised encoder matches or beats both PCA and the autoencoder on "
        f"{len(wins)} of {len(df.columns)} metrics" + (f": {', '.join(wins)}." if wins else "."))

st.subheader("Training loss")
h = run["history"]
if h.get("ssl"):
    loss = pd.DataFrame({"epoch": range(1, len(h["ssl"]["total"]) + 1), **{f"SSL {k}": v for k, v in h["ssl"].items()}})
    if h.get("ae"):
        loss["Autoencoder"] = pd.Series(h["ae"])
    st.plotly_chart(px.line(loss.melt("epoch"), x="epoch", y="value", color="variable"), use_container_width=True)
