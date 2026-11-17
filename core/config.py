"""Project-wide paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "physionet2012"
UPLOAD_DIR = DATA_DIR / "uploads"
FEATURES_CSV = DATA_DIR / "icu_features.csv"
LABELS_CSV = DATA_DIR / "icu_labels.csv"
DB_PATH = ROOT / "app.db"
MODELS_DIR = ROOT / "artifacts" / "models"

DEMO_DATASET_NAME = "PhysioNet 2012 ICU (Set-A)"
DISCLAIMER = (
    "Research prototype for a college project. It works on de-identified data and is "
    "**not** a clinical decision-support tool."
)

for _d in (DATA_DIR, UPLOAD_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
