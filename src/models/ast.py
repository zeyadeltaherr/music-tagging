"""Pretrained AST using official Kaldi Fbank features."""

import torch
import torch.nn as nn


class PretrainedAST(nn.Module):

    def __init__(
        self,
        n_tags,
        checkpoint="MIT/ast-finetuned-audioset-10-10-0.4593",
        freeze_backbone=True,
        target_length=1024,
        input_mean=-4.2677393,
        input_std=4.5689974,
        dropout=0.2,
    ):
        super().__init__()

        from transformers import ASTModel

        self.backbone = ASTModel.from_pretrained(checkpoint)

        self.target_length = int(target_length)
        self.input_mean = float(input_mean)
        self.input_std = float(input_std)
        self.freeze_backbone = bool(freeze_backbone)

        hidden = self.backbone.config.hidden_size

        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_tags)

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)

        if self.freeze_backbone:
            self.backbone.eval()

        return self

    def _prepare(self, fbank):
        # Input: (B, 1024, 128)
        x = fbank.float()

        x = (x - self.input_mean) / max(self.input_std, 1e-6)

        return x

    def embed(self, fbank):

        inputs = self._prepare(fbank)

        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(
                    input_values=inputs,
                    return_dict=True,
                )
        else:
            outputs = self.backbone(
                input_values=inputs,
                return_dict=True,
            )

        return outputs.last_hidden_state[:, 0, :]

    def classify(self, emb):
        return self.head(self.drop(emb))

    def forward(self, fbank):
        return self.classify(self.embed(fbank))