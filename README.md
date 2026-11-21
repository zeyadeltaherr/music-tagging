# Music Auto-Tagging and Similarity Search

**Course:** Deep Learning (AI403), Pharos University in Alexandria, Fall 2026
**Team:** <team name> — <member 1>, <member 2>, <member 3>, <member 4>
**Idea chosen:** #16 — Music auto-tagging and similarity search (audio → tags + neighbours)

## 1. Task and best result
Given a music clip, predict its tags (multi-label) and retrieve similar tracks using the learned embeddings.
Best model: **<fill in>** — macro ROC-AUC **<..>**, macro PR-AUC **<..>**, precision@10 **<..>** (mean ± SD over 3 seeds, test set).

## 2. Environment setup
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu132   # CUDA build, see pytorch.org
pip install -r requirements.txt
```

## 3. Dataset download and preprocessing
```powershell
python data/download.py --out C:\mtat_data
```
Raw data is NOT in the repository. Audio is converted to 16 kHz mono log-mel spectrograms
(`n_fft=512, hop=256, 128 mels`) inside `src/datasets/audio.py`. Splits: see `docs`/report (Section 6–7).

## 4. Training (one command per model; 3 seeds each)
```powershell
python -m src.train --config configs/cnn.yaml                    # classic CNN
python -m src.train --config configs/attn_cnn.yaml               # advanced: CNN + Transformer + attention pooling
python -m src.train --config configs/pretrained.yaml             # modern: pretrained model   <add when ready>
python -m src.train --config configs/ablation_no_attnpool.yaml   # ablations
python -m src.train --config configs/ablation_no_transformer.yaml
python -m src.train --config configs/ablation_no_augment.yaml
```
Baseline (MFCC + logistic regression): `python -m src.run_baseline`  <add when ready>

## 5. Evaluation
`src/evaluate.py` is shared by all models. `train.py` evaluates the best-validation checkpoint on the test set once and writes
`experiments/<run>.json` and `experiments/<name>_summary.json` (mean ± SD, per-tag results, cost).

## 6. Demo
```powershell
python -m src.build_index --ckpt checkpoints/attn_cnn_full_seed0.pt --split test --out checkpoints/index.npz
python demo/app.py --ckpt checkpoints/attn_cnn_full_seed0.pt --index checkpoints/index.npz
```

## 7. Hardware and runtimes
NVIDIA GeForce RTX 4050 Laptop GPU (6 GB), Windows, Python 3.13, PyTorch 2.15 (CUDA 13.2). Runtimes: <fill in per model>.
Large files (dataset, checkpoints): <shared drive link>.

## 8. Credits
- Dataset: MagnaTagATune — Law et al., "Evaluation of algorithms using games: the case of music annotation", ISMIR 2009. Hosted by City University London; Hugging Face mirror `confit/magnatagatune`.
- Libraries: PyTorch, torchaudio, librosa, scikit-learn, Gradio, Hugging Face Transformers.
- Pretrained model(s): <cite exactly which checkpoint and paper>.
- Our own code: models in `src/models/`, attention pooling layer, cosine kNN search (`src/similarity.py`), evaluation (`src/evaluate.py`).
