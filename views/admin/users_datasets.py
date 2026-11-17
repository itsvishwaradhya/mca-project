"""Manage user accounts and datasets."""
import pandas as pd
import streamlit as st

from core import auth, db
from core.ui import cached_dataset, header, require_admin

admin = require_admin()
header("Users & datasets")
tab_u, tab_d = st.tabs(["Users", "Datasets"])

with tab_u:
    users = db.q("SELECT id, username, role, active, created_at FROM users ORDER BY id")
    st.dataframe(pd.DataFrame(users), hide_index=True, use_container_width=True)

    st.subheader("Add user")
    with st.form("add_user", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        nu, npw = c1.text_input("Username"), c2.text_input("Password", type="password")
        role = c3.selectbox("Role", ["analyst", "admin"])
        if st.form_submit_button("Create"):
            ok, msg = auth.create_user(nu, npw, role)
            (st.success if ok else st.error)(msg)
            if ok:
                db.log(admin["username"], "create_user", f"{nu} ({role})")

    st.subheader("Change role / enable / disable")
    others = [u for u in users if u["username"] != admin["username"]]
    if others:
        target = st.selectbox("User", others, format_func=lambda u: f"{u['username']} ({u['role']}, "
                                                                    f"{'active' if u['active'] else 'disabled'})")
        c1, c2, c3 = st.columns(3)
        new_role = "admin" if target["role"] == "analyst" else "analyst"
        if c1.button(f"Make {new_role}"):
            db.ex("UPDATE users SET role=? WHERE id=?", (new_role, target["id"]))
            db.log(admin["username"], "change_role", f"{target['username']} -> {new_role}")
            st.rerun()
        if c2.button("Disable" if target["active"] else "Enable"):
            db.ex("UPDATE users SET active=? WHERE id=?", (0 if target["active"] else 1, target["id"]))
            db.log(admin["username"], "toggle_user", target["username"])
            st.rerun()
    else:
        st.caption("No other users yet.")
    st.caption("You cannot change your own role or disable yourself.")

with tab_d:
    ds = db.list_datasets()
    st.dataframe(pd.DataFrame([{k: d[k] for k in ("id", "name", "n_rows", "n_features", "created_by", "created_at")}
                               for d in ds]), hide_index=True, use_container_width=True)
    if ds:
        target = st.selectbox("Dataset", ds, format_func=lambda d: f"{d['id']} - {d['name']}")
        used = db.q1("SELECT COUNT(*) n FROM runs WHERE dataset_id=?", (target["id"],))["n"]
        if used:
            st.caption(f"{used} run(s) use this dataset; delete them first.")
        elif st.button("Delete dataset (cannot be undone)", type="secondary"):
            db.ex("DELETE FROM datasets WHERE id=?", (target["id"],))
            db.log(admin["username"], "delete_dataset", target["name"])
            cached_dataset.clear()
            st.rerun()
