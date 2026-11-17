"""Set the default flag threshold and review flagged patients as true / false positives."""
import pandas as pd
import streamlit as st

from core import db
from core.anomaly import flag
from core.ui import cached_dataset, cached_inference, header, require_admin

admin = require_admin()
header("Anomaly review", "Confirm or reject flagged patients; this gives a human-validated precision.")

st.subheader("Default threshold")
cur = float(db.get_setting("flag_percentile"))
new = st.slider("Flag the top X% by default (analysts can adjust)", 1.0, 20.0, cur, 0.5)
if new != cur and st.button("Save default"):
    db.set_setting("flag_percentile", str(new))
    db.log(admin["username"], "set_threshold", f"{new}%")
    st.rerun()

runs = db.list_runs("WHERE status='done'")
if not runs:
    st.info("No finished runs yet.")
    st.stop()

st.subheader("Review flagged patients")
run = st.selectbox("Run", runs, format_func=lambda r: f"{r['id']} - {r['name']}")
feats, labels, meta = cached_dataset(run["dataset_id"])
_, _, scores = cached_inference(run["id"], run["dataset_id"])
mask = flag(scores, new)
top_n = st.slider("Show the N most anomalous flagged patients", 5, 100, 20)
tbl = pd.DataFrame({"record_id": feats.index.astype(str), "score": scores.round(3)})[mask]
tbl = tbl.sort_values("score", ascending=False).head(top_n)

existing = {r["record_id"]: r["verdict"] for r in db.q("SELECT record_id, verdict FROM reviews WHERE run_id=?", (run["id"],))}
tbl["verdict"] = tbl["record_id"].map(lambda r: existing.get(r, "unreviewed"))
binary = meta.get("binary_label")
if labels is not None and binary in labels.columns:
    lab = pd.Series(labels[binary].values, index=feats.index.astype(str))
    tbl[f"actual {binary}"] = tbl["record_id"].map(lab)

edited = st.data_editor(
    tbl, hide_index=True, use_container_width=True, disabled=[c for c in tbl.columns if c != "verdict"],
    column_config={"verdict": st.column_config.SelectboxColumn(options=["unreviewed", "true_positive", "false_positive"])},
)
if st.button("Save reviews", type="primary"):
    n = 0
    for _, r in edited.iterrows():
        if r["verdict"] != "unreviewed" and existing.get(r["record_id"]) != r["verdict"]:
            db.ex("INSERT INTO reviews(run_id,record_id,score,verdict,reviewer,created_at) VALUES(?,?,?,?,?,?) "
                  "ON CONFLICT(run_id,record_id) DO UPDATE SET verdict=excluded.verdict, reviewer=excluded.reviewer, "
                  "created_at=excluded.created_at", (run["id"], r["record_id"], float(r["score"]), r["verdict"],
                                                      admin["username"], db.now()))
            n += 1
    db.log(admin["username"], "review_anomalies", f"run {run['id']}: {n} saved")
    st.success(f"Saved {n} review(s).")
    st.rerun()

st.subheader("Review summary for this run")
s = db.q1("SELECT SUM(verdict='true_positive') tp, SUM(verdict='false_positive') fp FROM reviews WHERE run_id=?", (run["id"],))
tp, fp = s["tp"] or 0, s["fp"] or 0
c1, c2, c3 = st.columns(3)
c1.metric("True positives", tp)
c2.metric("False positives", fp)
c3.metric("Precision (reviewed)", f"{100 * tp / (tp + fp):.0f}%" if tp + fp else "n/a")
