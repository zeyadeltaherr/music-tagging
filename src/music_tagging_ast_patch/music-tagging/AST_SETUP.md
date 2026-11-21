# AST pretrained model setup

This adds an Audio Spectrogram Transformer (AST) model to the existing training
pipeline. It uses the public AudioSet-pretrained checkpoint
`MIT/ast-finetuned-audioset-10-10-0.4593` and trains a 50-tag classification head.
The default configuration freezes the AST backbone to keep memory use manageable
on a 6-GB RTX 4050; this is a pretrained-feature baseline, not full AST fine-tuning.

## 1. Install dependency (inside the project's .venv)

```powershell
python -m pip install "transformers>=4.38,<5" "safetensors>=0.4"
```

The first run downloads the pretrained checkpoint from Hugging Face and caches it.
An internet connection is required for that first download.

## 2. Smoke test: one epoch, one seed

```powershell
python -m src.train --config configs/pretrained.yaml --epochs 1 --seeds 0
```

AST has a much larger input token sequence than the CNNs. Keep `batch_size: 1`
and `eval_batch_size: 1` on a 6-GB GPU. If CUDA runs out of memory, close other
GPU applications and retry. Do not change the data split.

## 3. Full run

Only after the smoke test finishes and the checkpoint/results are written:

```powershell
python -m src.train --config configs/pretrained.yaml --seeds 0 1 2
```

Outputs use the same `checkpoints/`, `experiments/`, validation selection, test
evaluation and metrics code as the CNN models.

## Method note

The project stores 128-bin torchaudio power-mel spectrograms in dB with 231-frame
training chunks. `PretrainedAST` converts these to AST's expected `(batch, time,
frequency)` shape, resizes time to 1024 frames, and applies the AudioSet feature
normalization constants. Since this is a frontend conversion rather than AST's
original waveform-to-fbank preprocessing, document it as an adaptation and
interpret results as an experimental pretrained baseline. The backbone is frozen
by default; only the 50-tag head is optimized. The existing CNN/Attention-CNN
training already satisfies the requirement that the team trains a model itself.
