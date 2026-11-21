"""Classic model: Short-chunk CNN on log-mel spectrograms.

A VGG-style stack of 3x3 conv blocks (Conv -> BatchNorm -> ReLU -> MaxPool),
followed by global max+average pooling and a small embedding layer.
Input : mel spectrogram, shape (B, n_mels, T)
Output: logits (B, n_tags); embeddings (B, emb_dim) via .embed()
Inspired by the short-chunk CNN family evaluated in Won et al. (2020).
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, pool=2):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(pool) if pool not in (None, 1, (1, 1)) else nn.Identity()

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))


class ShortChunkCNN(nn.Module):
    def __init__(self, n_tags, channels=64, emb_dim=256, dropout=0.3):
        super().__init__()
        c = channels
        chans = [1, c, c, 2 * c, 2 * c, 2 * c, 4 * c, 4 * c]
        pools = [2, 2, 2, 2, 2, 1, 1]  # 5 poolings -> works for T >= 32 frames
        self.input_bn = nn.BatchNorm2d(1)
        self.blocks = nn.Sequential(
            *[ConvBlock(chans[i], chans[i + 1], pools[i]) for i in range(7)]
        )
        self.proj = nn.Sequential(
            nn.Linear(2 * chans[-1], emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU()
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(emb_dim, n_tags)

    def embed(self, mel):
        x = self.input_bn(mel.unsqueeze(1))          # (B,1,F,T)
        x = self.blocks(x)                           # (B,C,F',T')
        x = torch.cat([x.amax(dim=(2, 3)), x.mean(dim=(2, 3))], dim=1)
        return self.proj(x)                          # (B,emb_dim)

    def classify(self, emb):
        return self.head(self.drop(emb))

    def forward(self, mel):
        return self.classify(self.embed(mel))
