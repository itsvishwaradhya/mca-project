"""Entry point: login / register, then role-based navigation (analyst vs admin)."""
import streamlit as st

from core import auth, db

st.set_page_config(page_title="ICU Patient-State Discovery", page_icon=":material/monitor_heart:", layout="wide")
db.init_db()

def analyst_pages(default: bool) -> list:
    """Analyst pages; only one page in the whole navigation may be the default."""
    return [
        st.Page("views/user/data_model.py", title="Data & Model", icon=":material/database:", default=default),
        st.Page("views/user/explore.py", title="Explore", icon=":material/bar_chart:"),
        st.Page("views/user/states.py", title="Patient states", icon=":material/hub:"),
        st.Page("views/user/at_risk.py", title="At-risk patients", icon=":material/warning:"),
        st.Page("views/user/compare.py", title="Compare methods", icon=":material/compare_arrows:"),
        st.Page("views/user/how_it_works.py", title="How it works", icon=":material/help:"),
    ]



ADMIN_PAGES = [
    st.Page("views/admin/dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True),
    st.Page("views/admin/users_datasets.py", title="Users & datasets", icon=":material/group:"),
    st.Page("views/admin/training.py", title="Training runs", icon=":material/model_training:"),
    st.Page("views/admin/experiments.py", title="Experiments", icon=":material/science:"),
    st.Page("views/admin/anomaly_review.py", title="Anomaly review", icon=":material/fact_check:"),
    st.Page("views/admin/audit.py", title="Audit log", icon=":material/history:"),
]


def login_screen() -> None:
    st.title(":material/monitor_heart: ICU Patient-State Discovery")
    st.caption("Self-supervised representation learning on unlabelled multivariate ICU data")
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        tab_in, tab_up = st.tabs(["Sign in", "Create analyst account"])
        with tab_in, st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in", type="primary", use_container_width=True):
                user = auth.authenticate(u, p)
                if user:
                    st.session_state["user"] = user
                    db.log(user["username"], "login")
                    st.rerun()
                db.log(u or None, "login_failed")
                st.error("Invalid username or password.")
        with tab_up, st.form("register"):
            nu = st.text_input("Choose a username")
            np1 = st.text_input("Password (min 8 characters)", type="password")
            np2 = st.text_input("Confirm password", type="password")
            if st.form_submit_button("Create account", use_container_width=True):
                if np1 != np2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = auth.create_user(nu, np1, "analyst")
                    (st.success if ok else st.error)(msg)
                    if ok:
                        db.log(nu, "register")


user = st.session_state.get("user")
if not user:
    login_screen()
    st.stop()

with st.sidebar:
    st.markdown(f"**{user['username']}**  \n{user['role'].title()}")
    if st.button("Sign out", icon=":material/logout:", use_container_width=True):
        db.log(user["username"], "logout")
        st.session_state.clear()
        st.rerun()

if user["role"] == "admin":
    pages = {"Admin": ADMIN_PAGES, "Analyst tools": analyst_pages(default=False)}
else:
    pages = {"Analyst": analyst_pages(default=True)}
st.navigation(pages).run()
