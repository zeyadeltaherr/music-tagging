"""Pretrained Audio Spectrogram Transformer (AST) feature extractor + task head.

Input is the project's cached log-mel tensor (B, 128, T). It is converted to
AST's (B, 1024, 128) input layout before passing through the pretrained backbone.
The default config freezes the backbone to fit 6-GB GPUs; the task head is trained.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PretrainedAST(nn.Module):
    def __init__(
        self,
        n_tags: int,
        checkpoint: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        freeze_backbone: bool = True,
        target_length: int = 1024,
        input_mean: float = -4.2677393,
        input_std: float = 4.5689974,
        dropout: float = 0.2,
    ):
        super().__init__()
        try:
            from transformers import ASTModel
        except ImportError as exc:
            raise ImportError(
                "AST requires transformers. Install it with: "
                "pip install 'transformers>=4.38,<5'"
            ) from exc

        self.backbone = ASTModel.from_pretrained(checkpoint)
        self.target_length = int(target_length)
        self.input_mean = float(input_mean)
        self.input_std = float(input_std)
        self.freeze_backbone = bool(freeze_backbone)
        hidden = int(self.backbone.config.hidden_size)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_tags)

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        # Keep a frozen pretrained backbone deterministic while training the head.
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _prepare(self, mel: torch.Tensor) -> torch.Tensor:
        # Project frontend is power-mel in dB, shape (B, 128, T).
        # Convert dB to natural-log energy, resize time to AST's expected length,
        # then apply the AST AudioSet feature normalization constants.
        x = mel.float().transpose(1, 2)  # (B, T, 128)
        x = x * (math.log(10.0) / 10.0)
        x = F.interpolate(
            x.transpose(1, 2), size=self.target_length,
            mode="linear", align_corners=False
        ).transpose(1, 2)
        x = (x - self.input_mean) / max(self.input_std, 1e-6)
        return x

    def embed(self, mel: torch.Tensor) -> torch.Tensor:
        inputs = self._prepare(mel)
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(input_values=inputs, return_dict=True)
        else:
            outputs = self.backbone(input_values=inputs, return_dict=True)
        # AST prepends CLS and distillation tokens; use CLS representation.
        return outputs.last_hidden_state[:, 0, :]

    def classify(self, emb: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(emb))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.classify(self.embed(mel))
