# Music Auto-Tagging and Similarity Search with Deep Learning
*Draft — sections marked **[TODO]** need results; **[VERIFY]** = check against the official source before submitting.*

## 1. Abstract  **[TODO — write last]**
One paragraph: task, dataset, the three deep approaches + baseline, best model and its macro ROC-AUC / PR-AUC (mean ± SD), the main finding on pretrained vs scratch and on the ablation, and the demo.

## 2. Introduction
Music streaming platforms host millions of tracks, and users find music through descriptive words ("calm", "guitar", "electronic") and through "more like this" recommendations. Producing these tags by hand does not scale, and tags written by users are sparse and inconsistent. Automatic music tagging uses deep learning to predict such tags directly from the audio signal, and the same learned representation can be reused to retrieve acoustically similar tracks. In this project we build a complete system: from a public dataset, through a baseline and three deep learning families, to a live demo that takes an audio file and returns predicted tags and similar tracks.

## 3. Problem Statement
- **Task:** multi-label music auto-tagging, plus content-based similarity search.
- **Input:** an audio clip (mono, 16 kHz), converted to a log-mel spectrogram.
- **Output:** (a) a probability for every tag in a fixed vocabulary; (b) an embedding vector used to rank the k most similar tracks by cosine similarity.
- **Hypotheses**
  - **H1 (pretraining):** a pretrained audio model (fine-tuned or linear-probed) reaches a higher macro PR-AUC than a CNN trained from scratch, under the same split, preprocessing and tuning budget.
  - **H2 (augmentation):** SpecAugment-style masking improves macro PR-AUC, with larger relative gains on low-support (rare) tags.
  - **H3 (temporal modelling):** adding a Transformer encoder and attention pooling over time improves macro ROC-AUC over the plain CNN, and attention pooling beats mean pooling.
  - **H4 (retrieval):** embeddings from trained deep models give higher precision@10 and Jaccard@10 than MFCC-statistics features.

## 4. Motivation
Tags drive search, playlist generation and recommendation, and they help cold-start tracks that have no listening history. The task is also a good testbed for deep learning: labels are noisy and strongly imbalanced, tags are correlated, audio is long-range temporal data, and the same model must serve both a classification and a retrieval goal.

## 5. Related Work  **[VERIFY all citations and add 1–2 sentences each]**
- Law et al., 2009 — MagnaTagATune dataset and the TagATune game (ISMIR).
- Choi et al., 2016 — Automatic tagging using deep convolutional neural networks (ISMIR).
- Pons & Serra, 2019 — musicnn: pre-trained convolutional neural networks for music audio tagging.
- Won et al., 2020 — Evaluation of CNN-based automatic music tagging models (SMC).
- Vaswani et al., 2017 — Attention is all you need.
- Park et al., 2019 — SpecAugment (Interspeech).
- Gong et al., 2021 — AST: Audio Spectrogram Transformer (Interspeech).
- Kong et al., 2020 — PANNs: large-scale pretrained audio neural networks.
- Spijkervet & Burgoyne, 2021 — CLMR: contrastive learning of musical representations.
- Li et al., 2023 — MERT: acoustic music understanding model with large-scale self-supervised training.

## 6. Dataset  **[VERIFY all numbers and the licence from the official page]**
MagnaTagATune (Law et al., 2009), hosted by City University London (mirg.city.ac.uk). Short audio clips of music from the Magnatune label, with tags collected from players of the TagATune game. We use the standard top-50 tag vocabulary and the standard train/validation/test split from the files `train_gt_mtt.tsv`, `val_gt_mtt.tsv`, `test_gt_mtt.tsv`. Leakage check: **[TODO — state how we verified that clips of the same song do not appear in different splits]**. The test set is used only for final evaluation.

**Table 1 — Dataset comparison**
| Dataset | Source | Size (clips) | Format | Classes (tags) | Licence | Role |
|---|---|---|---|---|---|---|
| MagnaTagATune | mirg.city.ac.uk | **[TODO]** | mp3, 16 kHz | 50 | **[VERIFY]** | train / val / test |

## 7. Data Preprocessing  **[TODO — confirm after the loader is written]**
Resample to 16 kHz mono; random fixed-length chunks (≈3.69 s) for training, multiple chunks averaged at test time; log-mel spectrogram (n_fft 512, hop 256, 128 mels); input batch normalisation inside the models. Augmentation: frequency and time masking.

## 8. Methodology
Every model shares the same data split, preprocessing, optimiser family, tuning budget and evaluation code (`src/evaluate.py`). Each final configuration is run with 3 random seeds and reported as mean ± SD. Model selection uses validation macro PR-AUC; the test set is evaluated once per run.

## 9. Baseline
MFCC (20 coefficients) mean and standard deviation per clip → standardisation → one-vs-rest logistic regression; the regularisation strength C is chosen on the validation set. **[TODO results]**

## 10. Proposed / Advanced Models
- **Classic — Short-chunk CNN** (`src/models/cnn.py`): 7 convolutional blocks, global max+average pooling, 256-d embedding.
- **Modern — pretrained model** **[TODO: name the checkpoint, e.g. AST or PANNs; linear probe vs fine-tune]**.
- **Advanced — CNN + Transformer + attention pooling** (`src/models/attention_cnn.py`): convolutional stem, sinusoidal positional encoding, Transformer encoder, and our own additive attention-pooling layer.
- **Our own components:** the attention-pooling layer, the cosine-similarity kNN search (`src/similarity.py`), and the evaluation metrics.

**Table 2 — Model architecture comparison**
| Model | Family | Input | Main layers | Parameters | Pretrained? |
|---|---|---|---|---|---|
| MFCC + LR | Classical ML | MFCC stats | Logistic regression | **[TODO]** | No |
| Short-chunk CNN | Classic | log-mel | 7 conv blocks | **[TODO]** | No |
| **[pretrained]** | Modern | **[TODO]** | **[TODO]** | **[TODO]** | Yes |
| CNN + Transformer + attn pool | Advanced | log-mel | CNN stem, 2 Transformer layers, attention pooling | **[TODO]** | No |

## 11. Experimental Setup
Hardware: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB). Framework: PyTorch with mixed precision. Seeds: 0, 1, 2. Metrics: macro ROC-AUC (primary for comparability with the literature), macro PR-AUC (important under class imbalance), macro-F1 at threshold 0.5, per-tag ROC/PR-AUC, 95 % bootstrap CI on test ROC-AUC, retrieval precision@10 and Jaccard@10.

## 12. Hyperparameters
**Table 3 — Hyperparameter configuration**
| Model | Optimizer | LR | Batch size | Epochs | Regularization | Loss | Augmentation |
|---|---|---|---|---|---|---|---|
| Short-chunk CNN | AdamW | 1e-3 | 32 | ≤20 (early stop) | weight decay 1e-4, dropout 0.3 | BCE | SpecAugment |
| CNN + Transformer | AdamW | 5e-4 | 32 | ≤20 (early stop) | weight decay 1e-4, dropout 0.2 | BCE | SpecAugment |
| **[pretrained]** | **[TODO]** | | | | | | |

Hyperparameter study **[TODO]**: learning rate, batch size, dropout (same budget for every model).
Data-strategy study **[TODO]**: with vs without SpecAugment (and/or class weighting).

## 13. Results
**Table 4 — Main experimental results (test set, mean ± SD over 3 seeds)**
| Model | ROC-AUC (macro) | PR-AUC (macro) | F1 (macro) | precision@10 | Jaccard@10 |
|---|---|---|---|---|---|
| MFCC + LR | | | | | |
| Short-chunk CNN | | | | | |
| **[pretrained]** | | | | | |
| CNN + Transformer + attn pool | | | | | |

Pretrained vs scratch comparison **[TODO]**. Per-tag results: figure from `per_tag` in `experiments/*.json`.

## 14. Model Comparison
**[TODO]** Same-protocol comparison, then a verdict on each hypothesis (H1–H4): supported / partly supported / rejected, with the numbers.

## 15. Ablation Study
**Table 5 — Ablation study**
| Component removed | Macro ROC-AUC | Macro PR-AUC | Change vs full model |
|---|---|---|---|
| (full model) | | | — |
| Attention pooling → mean pooling | | | |
| Transformer encoder | | | |
| SpecAugment | | | |
| Pretraining (pretrained model only) | | | |

## 16. Error Analysis
**Table 6 — Error analysis**
| Error type | Count | Example (clip id, tag) | Likely cause |
|---|---|---|---|
| Rare tags (low support) | | | |
| Confused tag pairs (e.g. similar genres) | | | |
| Instrument tags hidden by mixture | | | |
| Retrieval: wrong neighbours | | | |

## 17. Discussion  **[TODO]**
Why one model beats another; what the attention weights show; effect of label noise and imbalance.

## 18. Limitations
Noisy, subjective tags; small, single-label-source dataset from one label (genre/style bias); short clips; limited tuning budget and compute (single 6 GB GPU); few seeds.

## 19. Ethical Considerations
Tags reflect the cultural biases of the annotators and of the label's catalogue; models may under-serve non-Western or under-represented genres. Dataset licence (non-commercial terms **[VERIFY]**) restricts commercial use. No personal data is used; the demo processes uploaded audio only in memory.

## 20. Conclusion  **[TODO]**

## 21. Future Work
Larger datasets (e.g. MTG-Jamendo), self-supervised contrastive pretraining, tag-correlation-aware losses, hybrid retrieval with metadata, evaluation with human listening tests.

## 22. References  **[TODO — full BibTeX/IEEE entries after verifying each]**

---
**Table 7 — Computational cost**
| Model | Parameters | Size (MB) | Train time | Inference (ms/clip) | Memory (peak GPU) | Hardware |
|---|---|---|---|---|---|---|
| (filled automatically from `experiments/*_summary.json` → `cost_first_seed`) | | | | | | |
