"""Advanced model: CNN stem + Transformer encoder + attention pooling.

Our own component: `AttentionPooling` (learned, additive attention over time),
written from scratch rather than imported from a library.

Flags for the ablation study:
  use_transformer=False     -> remove the Transformer encoder
  use_attention_pool=False  -> replace attention pooling by plain mean pooling
"""
import math
import torch
import torch.nn as nn

from .cnn import ConvBlock


class AttentionPooling(nn.Module):
    """alpha_t = softmax_t( v^T tanh(W h_t) ),  output = sum_t alpha_t * h_t"""

    def __init__(self, d_model, d_attn=128):
        super().__init__()
        self.w = nn.Linear(d_model, d_attn)
        self.v = nn.Linear(d_attn, 1, bias=False)

    def forward(self, h):                       # h: (B, T, D)
        scores = self.v(torch.tanh(self.w(h))).squeeze(-1)   # (B, T)
        alpha = torch.softmax(scores.float(), dim=1).to(h.dtype)
        pooled = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)  # (B, D)
        return pooled, alpha


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):                       # (B, T, D)
        return x + self.pe[:, : x.size(1)].to(x.dtype)


class AttentionCNN(nn.Module):
    def __init__(self, n_tags, channels=32, d_model=128, n_heads=4, n_layers=2,
                 emb_dim=256, dropout=0.2, use_transformer=True, use_attention_pool=True):
        super().__init__()
        c = channels
        self.use_transformer = use_transformer
        self.use_attention_pool = use_attention_pool
        self.input_bn = nn.BatchNorm2d(1)
        # frequency 128 -> 8, time / 8
        self.stem = nn.Sequential(
            ConvBlock(1, c, (2, 2)),
            ConvBlock(c, 2 * c, (2, 2)),
            ConvBlock(2 * c, 2 * c, (2, 2)),
            ConvBlock(2 * c, 4 * c, (2, 1)),
        )
        self.stem_c = 4 * c
        self.to_model = nn.LazyLinear(d_model)   # (C * F') -> d_model, F' inferred
        self.pos = SinusoidalPositionalEncoding(d_model)
        if use_transformer:
            layer = nn.TransformerEncoderLayer(
                d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout,
                batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.pool = AttentionPooling(d_model) if use_attention_pool else None
        self.proj = nn.Sequential(nn.Linear(d_model, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU())
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(emb_dim, n_tags)
        self.last_attention = None               # (B, T) weights from the last forward

    def tokens(self, mel):
        x = self.input_bn(mel.unsqueeze(1))      # (B,1,F,T)
        x = self.stem(x)                         # (B,C,F',T')
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)   # (B,T',C*F')
        x = self.pos(self.to_model(x))
        if self.use_transformer:
            x = self.encoder(x)
        return x                                 # (B,T',D)

    def embed(self, mel):
        h = self.tokens(mel)
        if self.pool is not None:
            pooled, alpha = self.pool(h)
            self.last_attention = alpha.detach()
        else:
            pooled = h.mean(dim=1)
            self.last_attention = None
        return self.proj(pooled)

    def classify(self, emb):
        return self.head(self.drop(emb))

    def forward(self, mel):
        return self.classify(self.embed(mel))
