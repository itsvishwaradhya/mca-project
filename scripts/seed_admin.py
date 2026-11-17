"""Create the demo admin + analyst accounts and register the demo dataset.

Reads DEMO_* variables from .env (falls back to .env.example). Safe to re-run.
Usage: python scripts/seed_admin.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import auth, db  # noqa: E402
from core.config import DEMO_DATASET_NAME, FEATURES_CSV, LABELS_CSV  # noqa: E402


def load_env() -> None:
    for name in (".env", ".env.example"):
        p = ROOT / name
        if p.exists():
            for line in p.read_text().splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
            return


def register_demo_dataset() -> None:
    import pandas as pd
    if not FEATURES_CSV.exists():
        print("Demo dataset not built yet (run: python -m core.ingest). Skipping dataset registration.")
        return
    if any(d["name"] == DEMO_DATASET_NAME for d in db.list_datasets()):
        return
    feats = pd.read_csv(FEATURES_CSV, index_col=0)
    meta = {"binary_label": "In_hospital_death", "cat_label": "ICUType", "score_cols": ["SAPS_I", "SOFA"],
            "id_name": "RecordID", "demo": True}
    db.add_dataset(DEMO_DATASET_NAME, FEATURES_CSV, LABELS_CSV, meta, len(feats), feats.shape[1], "system")
    print("Registered demo dataset.")


def main() -> None:
    load_env()
    db.init_db()
    for role, ukey, pkey in (("admin", "DEMO_ADMIN_USER", "DEMO_ADMIN_PASSWORD"),
                             ("analyst", "DEMO_ANALYST_USER", "DEMO_ANALYST_PASSWORD")):
        ok, msg = auth.create_user(os.environ[ukey], os.environ[pkey], role)
        print(f"{role}: {msg}")
    register_demo_dataset()


if __name__ == "__main__":
    main()
