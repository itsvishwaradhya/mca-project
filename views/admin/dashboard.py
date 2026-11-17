"""Admin overview: counts, run status, recent activity."""
import pandas as pd
import streamlit as st

from core import db
from core.ui import header, require_admin

require_admin()
header("Admin dashboard", "System overview.")

n = lambda sql: db.q1(sql)["n"]  # noqa: E731
c = st.columns(5)
c[0].metric("Users", n("SELECT COUNT(*) n FROM users WHERE active=1"))
c[1].metric("Datasets", n("SELECT COUNT(*) n FROM datasets"))
c[2].metric("Runs finished", n("SELECT COUNT(*) n FROM runs WHERE status='done'"))
c[3].metric("Published models", n("SELECT COUNT(*) n FROM runs WHERE published=1"))
c[4].metric("Anomalies reviewed", n("SELECT COUNT(*) n FROM reviews"))

left, right = st.columns(2)
with left:
    st.subheader("Recent runs")
    runs = db.q("SELECT id, name, status, ROUND(progress*100) AS pct, published, created_by, created_at "
                "FROM runs ORDER BY id DESC LIMIT 8")
    if runs:
        st.dataframe(pd.DataFrame(runs), hide_index=True, use_container_width=True)
    else:
        st.caption("No runs yet.")
with right:
    st.subheader("Recent activity")
    st.dataframe(pd.DataFrame(db.q("SELECT ts, username, action, detail FROM audit ORDER BY id DESC LIMIT 8")),
                 hide_index=True, use_container_width=True)

rev = db.q1("SELECT SUM(verdict='true_positive') tp, COUNT(*) n FROM reviews")
if rev["n"]:
    st.subheader("Reviewed anomaly precision")
    st.metric("True positives among reviewed flags", f"{100 * rev['tp'] / rev['n']:.0f}%", f"{rev['n']} reviewed")
