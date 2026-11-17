"""Searchable audit log of logins, uploads, runs and admin actions."""
import pandas as pd
import streamlit as st

from core import db
from core.ui import header, require_admin

require_admin()
header("Audit log")

rows = pd.DataFrame(db.q("SELECT ts, username, action, detail FROM audit ORDER BY id DESC LIMIT 2000"))
if rows.empty:
    st.info("Nothing logged yet.")
    st.stop()
c1, c2 = st.columns(2)
actions = c1.multiselect("Action", sorted(rows["action"].unique()))
who = c2.multiselect("User", sorted(rows["username"].dropna().unique()))
if actions:
    rows = rows[rows["action"].isin(actions)]
if who:
    rows = rows[rows["username"].isin(who)]
st.dataframe(rows, hide_index=True, use_container_width=True)
st.download_button("Download CSV", rows.to_csv(index=False), "audit_log.csv", "text/csv")
