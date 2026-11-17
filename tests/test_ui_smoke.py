"""Headless smoke tests of the Streamlit pages (needs the demo dataset + at least one finished run)."""
import pytest
from streamlit.testing.v1 import AppTest

from core import db
from core.config import FEATURES_CSV, ROOT

pytestmark = pytest.mark.skipif(not FEATURES_CSV.exists(), reason="demo data not built")

ANALYST = {"id": 0, "username": "smoke_analyst", "role": "analyst"}
ADMIN = {"id": 0, "username": "smoke_admin", "role": "admin"}


def run_page(path: str, user: dict, **state):
    at = AppTest.from_file(str(ROOT / path), default_timeout=120)
    at.session_state["user"] = user
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


def finished_run():
    runs = db.list_runs("WHERE status='done'")
    if not runs:
        pytest.skip("no finished run")
    return runs[0]


@pytest.mark.parametrize("page", ["views/user/data_model.py", "views/user/how_it_works.py"])
def test_user_pages_load(page):
    db.init_db()
    assert not run_page(page, ANALYST).exception


def test_model_pages_load_with_context():
    r = finished_run()
    for page in ("views/user/explore.py", "views/user/compare.py", "views/user/at_risk.py"):
        at = run_page(page, ANALYST, dataset_id=r["dataset_id"], run_id=r["id"])
        assert not at.exception, (page, at.exception)


@pytest.mark.parametrize("page", ["dashboard", "users_datasets", "experiments", "anomaly_review", "audit", "training"])
def test_admin_pages_load(page):
    assert not run_page(f"views/admin/{page}.py", ADMIN).exception


def test_analyst_blocked_from_admin_pages():
    at = run_page("views/admin/dashboard.py", ANALYST)
    assert any("Admin access required" in e.value for e in at.error)


def test_pages_stop_without_context():
    at = run_page("views/user/states.py", ANALYST)
    assert any("Data & Model" in i.value for i in at.info)
