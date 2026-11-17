# ICU Patient-State Discovery (Self-Supervised Learning)

MCA project: *Self-Supervised Learning for Representation Discovery in Unlabelled Multivariate Data*, applied to
healthcare. An encoder trained with **contrastive + masked-reconstruction** objectives learns patient representations
from ICU measurements **without using outcome labels**. Labels (mortality, ICU type, SAPS-I, SOFA) are used only to
evaluate the result. The approach is compared with **PCA** and a **plain autoencoder**.

Data: [PhysioNet/CinC Challenge 2012](https://physionet.org/content/challenge-2012/1.0.0/) (open access, de-identified).
This is a research prototype, not a clinical decision-support tool.

## Setup (Windows, from this folder)
```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python scripts/download_data.py      # Set-A + outcomes (~8 MB)
.venv/Scripts/python -m core.ingest                # one feature row per ICU stay
.venv/Scripts/python scripts/seed_admin.py         # demo accounts + register dataset (see .env.example)
.venv/Scripts/python -m streamlit run app.py
```
Demo account names/passwords are in `.env.example` (copy to `.env` and change them before sharing).

## Interfaces
**Analyst (user)**: Data & Model (pick/upload CSV, quick-train) -> Explore -> Patient states (UMAP + state profiles)
-> At-risk patients (ranked anomalies) -> Compare methods -> How it works.

**Admin**: Dashboard, Users & datasets, Training runs (hyperparameters, background training, live loss, publish model),
Experiments (cross-run comparison), Anomaly review (true/false positives, default threshold), Audit log.

## Mapping to the synopsis objectives
| Objective | Where |
|---|---|
| Study SSL for unlabelled multivariate data | `core/train.py`, *How it works* page |
| Preprocess, augmentation and masking | `core/preprocess.py`, `core/augment.py` |
| Encoder with contrastive + reconstruction pretext tasks | `core/models.py`, `core/train.py` |
| Clustering and dimensionality reduction on representations | *Patient states* page (KMeans + UMAP) |
| Compare with PCA and autoencoder | `core/baselines.py`, *Compare methods* page |
| Evaluate with clustering metrics + linear probe (labels only for evaluation) | `core/evaluate.py` |

## Tests
```bash
.venv/Scripts/python -m pytest -q
```
