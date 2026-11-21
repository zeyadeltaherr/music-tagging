# Suggested work split (4–5 members) — everyone must understand the WHOLE system for the viva
| Role | Owner | Tasks |
|---|---|---|
| A — Data and baseline | <name> | `data/download.py`, split check (no leakage), dataset loader `src/datasets/mtat.py`, MFCC baseline, Table 1 |
| B — Classic model and pipeline | <name> | run `cnn.yaml` (3 seeds), hyperparameter study, SpecAugment study, Tables 3–4 rows |
| C — Modern (pretrained) model | <name> | AST/PANNs wrapper, linear probe vs fine-tune, pretrained-vs-scratch comparison |
| D — Advanced model and ablation | <name> | `attn_cnn.yaml` (3 seeds), 3 ablation configs, Table 5, attention-weight figures |
| E — Demo, similarity and docs | <name> | `build_index.py`, Gradio demo, README, error analysis (Table 6), slides |
Report: each owner writes the section for their part; one person merges and proof-reads.
Peer evaluation and logs: each student keeps `TeamName_StudentID_Log.pdf`, co-signed by a teammate.
