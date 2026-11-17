"""Explore the raw multivariate data: distributions, correlations, missingness."""
import plotly.express as px
import streamlit as st

from core.ui import cached_dataset, context, header

header("Explore", "Look at the data before any learning happens.")
ds_id, _ = context(need_model=False)
feats, labels, meta = cached_dataset(ds_id)

tab_dist, tab_corr, tab_miss = st.tabs(["Distributions", "Correlations", "Missingness"])

with tab_dist:
    cols = list(feats.columns)
    default = cols.index("HR_mean") if "HR_mean" in cols else 0
    var = st.selectbox("Feature", cols, index=default)
    color_opts = ["(none)"] + [c for c in (labels.columns if labels is not None else [])
                               if labels[c].nunique() <= 20]
    color = st.selectbox("Colour by (evaluation label)", color_opts)
    df = feats[[var]].copy()
    if color != "(none)":
        df[color] = labels[color].astype(str).values
    fig = px.histogram(df, x=var, color=None if color == "(none)" else color, nbins=50, barmode="overlay",
                       opacity=0.7, marginal="box")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(feats[var].describe().to_frame().T, use_container_width=True)

with tab_corr:
    var_rank = feats.std(numeric_only=True).div(feats.mean(numeric_only=True).abs() + 1e-9).sort_values(ascending=False)
    n = st.slider("Number of features (highest variation first)", 5, 40, 20)
    top = list(var_rank.head(n).index)
    st.plotly_chart(px.imshow(feats[top].corr(), color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto"),
                    use_container_width=True)

with tab_miss:
    miss = feats.isna().mean().mul(100).sort_values(ascending=False).head(30).reset_index()
    miss.columns = ["feature", "missing_%"]
    st.plotly_chart(px.bar(miss, x="missing_%", y="feature", orientation="h").update_yaxes(autorange="reversed"),
                    use_container_width=True)
    st.caption("In ICU data a value is often missing because the test was never ordered, "
               "which is itself clinically informative. The `_count` features capture this.")
