"""Turn raw PhysioNet 2012 patient files into one fixed-length feature row per ICU stay.

Each raw file holds irregular (Time, Parameter, Value) records for the first 48 h.
Per variable we keep min / max / mean / last / count. Outcomes, ICU type and severity
scores go to a separate *evaluation-only* table and never reach the encoder.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FEATURES_CSV, LABELS_CSV, RAW_DIR

DESCRIPTORS = ["Age", "Gender", "Height", "Weight"]  # used as features
EVAL_ONLY = ["ICUType"]  # general descriptor kept only for evaluation
TS_VARS = [
    "Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN", "Cholesterol", "Creatinine", "DiasABP",
    "FiO2", "GCS", "Glucose", "HCO3", "HCT", "HR", "K", "Lactate", "MAP", "MechVent", "Mg", "Na",
    "NIDiasABP", "NIMAP", "NISysABP", "PaCO2", "PaO2", "pH", "Platelets", "RespRate", "SaO2",
    "SysABP", "Temp", "TroponinI", "TroponinT", "Urine", "WBC",
]
STATS = ["min", "max", "mean", "last", "count"]
MISSING = -1  # PhysioNet encodes "not recorded" as -1


def parse_patient(path: Path) -> dict:
    df = pd.read_csv(path)
    rec = {}
    gen = df[df["Time"] == "00:00"]
    for name in ["RecordID", *DESCRIPTORS, *EVAL_ONLY]:
        v = gen.loc[gen["Parameter"] == name, "Value"]
        val = float(v.iloc[0]) if len(v) else np.nan
        rec[name] = np.nan if (name != "RecordID" and val == MISSING) else val
    ts = df[df["Parameter"].isin(TS_VARS) & (df["Value"] != MISSING)].copy()
    ts["minutes"] = ts["Time"].str.split(":").map(lambda t: int(t[0]) * 60 + int(t[1]))
    ts = ts.sort_values("minutes")
    g = ts.groupby("Parameter")["Value"]
    agg = pd.DataFrame({"min": g.min(), "max": g.max(), "mean": g.mean(), "last": g.last(), "count": g.count()})
    for var in TS_VARS:
        for s in STATS:
            if var in agg.index:
                rec[f"{var}_{s}"] = float(agg.loc[var, s])
            else:
                rec[f"{var}_{s}"] = 0.0 if s == "count" else np.nan
    return rec


def build_dataset(raw_dir: Path = RAW_DIR, set_name: str = "set-a", outcomes_file: str = "Outcomes-a.txt"):
    files = sorted((raw_dir / set_name).glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No patient files in {raw_dir/set_name}. Run scripts/download_data.py first.")
    rows = [parse_patient(f) for f in files]
    df = pd.DataFrame(rows)
    df["RecordID"] = df["RecordID"].astype(int)
    out = pd.read_csv(raw_dir / outcomes_file)
    labels = df[["RecordID", *EVAL_ONLY]].merge(out, on="RecordID", how="left")
    labels = labels.rename(columns={"In-hospital_death": "In_hospital_death", "SAPS-I": "SAPS_I"})
    labels["ICUType"] = labels["ICUType"].map({1: "Coronary", 2: "Cardiac surgery", 3: "Medical", 4: "Surgical"})
    feats = df.drop(columns=EVAL_ONLY).set_index("RecordID")
    labels = labels.set_index("RecordID").loc[feats.index]
    return feats, labels


def main() -> None:
    feats, labels = build_dataset()
    feats.to_csv(FEATURES_CSV)
    labels.to_csv(LABELS_CSV)
    print(f"features {feats.shape} -> {FEATURES_CSV}\nlabels   {labels.shape} -> {LABELS_CSV}")


if __name__ == "__main__":
    main()
