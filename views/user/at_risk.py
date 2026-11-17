"""Anomaly / at-risk patient ranking from the learned representation."""
import pandas as pd
import plotly.express as px
import streamlit as st

from core import db
from core.anomaly import flag, top_deviations
from core.ui import cached_dataset, cached_inference, cached_model, context, header

header("At-risk patients", "Patients whose learned representation is unusual compared with the rest of the cohort.")
ds_id, run_id = context()
feats, labels, meta = cached_dataset(ds_id)
X, Z, scores = cached_inference(run_id, ds_id)
names = cached_model(run_id).prep.feature_names

default_pct = float(db.get_setting("flag_percentile"))
pct = st.slider("Flag the top X% most unusual patients", 1.0, 20.0, default_pct, 0.5,
                help=f"Admin default is {default_pct}%.")
mask = flag(scores, pct)

tbl = pd.DataFrame({"id": feats.index.astype(str), "anomaly_score": scores.round(3), "flagged": mask})
binary = meta.get("binary_label")
if labels is not None and binary in (labels.columns if labels is not None else []):
    tbl[binary] = labels[binary].values

c1, c2, c3 = st.columns(3)
c1.metric("Patients flagged", int(mask.sum()))
c2.metric("Score cut-off", f"{scores[mask].min():.2f}")
if binary and binary in tbl:
    base = pd.to_numeric(tbl[binary], errors="coerce").mean() * 100
    hit = pd.to_numeric(tbl.loc[tbl.flagged, binary], errors="coerce").mean() * 100
    c3.metric(f"{binary} rate in flagged", f"{hit:.1f}%", f"{hit - base:+.1f} pts vs cohort {base:.1f}%")

st.plotly_chart(px.histogram(tbl, x="anomaly_score", color="flagged", nbins=50, barmode="overlay", opacity=0.8),
                use_container_width=True)

st.subheader("Ranked list")
ranked = (tbl[tbl.flagged].drop(columns="flagged").sort_values("anomaly_score", ascending=False)
          .reset_index(drop=True))
sel = st.dataframe(ranked, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row")
st.download_button("Download flagged patients (CSV)", ranked.to_csv(index=False), "flagged_patients.csv", "text/csv")

rows = sel.selection.rows if sel else []
if rows:
    pid = ranked.loc[rows[0], "id"]
    i = int((tbl["id"] == pid).idxmax())
    st.subheader(f"Patient {pid}")
    left, right = st.columns(2)
    left.write("Features that deviate most from the cohort (z-scores):")
    left.dataframe(top_deviations(X, names, i, 8), hide_index=True, use_container_width=True)
    if labels is not None:
        right.write("Evaluation labels (not used by the model):")
        right.dataframe(labels.iloc[[i]].T.rename(columns={labels.index[i]: "value"}), use_container_width=True)
st.caption("Flags are a ranking aid for investigation, not a diagnosis.")
