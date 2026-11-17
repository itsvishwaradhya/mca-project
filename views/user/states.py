"""Discovered patient states: 2D map of the learned representation + cluster profiles."""
import pandas as pd
import plotly.express as px
import streamlit as st

from core.ui import (cached_best_k, cached_clusters, cached_dataset, cached_projection, context, db, header,
                     profile_for)

header("Patient states", "Groups of similar patients discovered without using any outcome labels.")
ds_id, run_id = context()
_, labels, meta = cached_dataset(ds_id)
run = db.get_run(run_id)

default_k = int(run["config"].get("n_clusters", 5))
c1, c2 = st.columns([3, 1])
if c2.button("Suggest k (silhouette)"):
    best, scores = cached_best_k(run_id, ds_id)
    st.session_state["k_states"] = best
    c2.caption(f"Best k = {best}")
k = c1.slider("Number of states (k)", 2, 10, st.session_state.get("k_states", default_k), key="k_states")

clusters = cached_clusters(run_id, ds_id, k)
xy = cached_projection(run_id, ds_id)

color_opts = ["Discovered state"]
if labels is not None:
    color_opts += [c for c in labels.columns if labels[c].nunique() <= 20]
color_by = st.radio("Colour the map by", color_opts, horizontal=True)

df = labels.copy() if labels is not None else None
plot = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "state": clusters.astype(str)})
if df is not None:
    for c in color_opts[1:]:
        plot[c] = df[c].astype(str).values
plot["id"] = cached_dataset(ds_id)[0].index.astype(str)
fig = px.scatter(plot, x="x", y="y", color="state" if color_by == "Discovered state" else color_by,
                 hover_data=["id"], opacity=0.75, height=520,
                 category_orders={"state": [str(i) for i in range(k)]})
fig.update_traces(marker_size=5)
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
st.plotly_chart(fig, use_container_width=True)
st.caption("Each dot is a patient. Nearby dots have similar learned representations. "
           "Colouring by an outcome label is only a sanity check: the model never saw it.")

st.subheader("What defines each state")
prof = profile_for(run_id, ds_id, k)
st.dataframe(prof, use_container_width=True, hide_index=True)

rate_col = next((c for c in prof.columns if c.endswith("_rate_%")), None)
if rate_col:
    st.plotly_chart(px.bar(prof, x="state", y=rate_col, text=rate_col, title=f"{rate_col} per discovered state")
                    .update_layout(xaxis_type="category"), use_container_width=True)
    spread = prof[rate_col].max() - prof[rate_col].min()
    st.info(f"Outcome rate varies by {spread:.0f} percentage points across states. "
            "A wide spread means the unsupervised states separate low- and high-risk patients.")
st.download_button("Download state assignments (CSV)",
                   plot[["id", "state"]].to_csv(index=False), "patient_states.csv", "text/csv")
