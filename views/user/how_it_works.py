"""Plain-language explainer of the pipeline (maps to the synopsis objectives)."""
import streamlit as st

from core.ui import current_user, disclaimer, header

current_user()
header("How it works", "From raw ICU measurements to discovered patient states, without using outcome labels.")
disclaimer()

st.graphviz_chart("""
digraph G {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Helvetica"];
  raw [label="Raw ICU records\\n(36 vitals/labs, 48 h)"];
  feat [label="One row per patient\\nmin/max/mean/last/count"];
  prep [label="Impute + scale"];
  aug [label="Two augmented views\\nnoise + feature masking"];
  enc [label="Encoder (MLP)", fillcolor="#dbe7fb"];
  con [label="Contrastive loss\\n(same patient = similar)"];
  rec [label="Reconstruction loss\\n(fill masked features)"];
  emb [label="Patient embedding", fillcolor="#c6f0d6"];
  out [label="States  |  At-risk ranking  |  Metrics"];
  raw -> feat -> prep -> aug -> enc; enc -> con; enc -> rec; enc -> emb -> out;
}
""")

st.markdown("""
### The idea
1. **No labels in training.** Survival, ICU type and severity scores are kept aside and used *only* to check the result.
2. **Make the model learn by itself (pretext tasks).**
   - *Contrastive*: take a patient, corrupt the record twice in different ways (noise and hidden measurements).
     The encoder must give the two corrupted copies similar embeddings and different patients different ones.
   - *Reconstruction*: hide some measurements and make the model fill them in from the rest.
3. **Use the embedding.** Cluster it to find patient states, score how unusual each patient is, and measure quality.

### Why compare with PCA and an autoencoder?
They are the standard unsupervised alternatives. If the self-supervised embedding does not beat them, the extra
machinery is not worth it, so the **Compare methods** page reports all of them honestly.

### Metrics
| Metric | Meaning |
|---|---|
| Silhouette / Davies-Bouldin | How compact and well separated the clusters are (no labels needed) |
| NMI / ARI vs ICU type | Agreement of discovered states with a known grouping (label used only here) |
| Linear-probe AUROC | A simple classifier on the frozen embedding predicts mortality: does the embedding hold risk information? |
| Anomaly AUROC | Do high-anomaly patients tend to be the ones who died? |

### Mapping to the project objectives
Pre-processing and augmentation → *Data & Model*; encoder with contrastive + reconstruction tasks → *Training runs (admin)*;
structure discovery → *Patient states* and *At-risk patients*; baseline comparison and evaluation → *Compare methods*.
""")
