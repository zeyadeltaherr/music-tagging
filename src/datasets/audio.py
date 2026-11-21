"""Audio -> log-mel front-end shared by training, indexing and the demo."""
import numpy as np
import torch
import torch.nn as nn

SR = 16000
N_FFT = 512
HOP = 256
N_MELS = 128
CLIP_SAMPLES = 59049          # ~3.69 s chunks


def load_wav(path, sr=SR):
    import librosa
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav.astype(np.float32)


def make_chunks(wav, clip_samples=CLIP_SAMPLES, max_chunks=10):
    """Split a waveform into equal chunks (last one zero-padded)."""
    if len(wav) < clip_samples:
        wav = np.pad(wav, (0, clip_samples - len(wav)))
    n = min(max_chunks, max(1, int(np.ceil(len(wav) / clip_samples))))
    chunks = []
    for i in range(n):
        c = wav[i * clip_samples:(i + 1) * clip_samples]
        if len(c) < clip_samples:
            c = np.pad(c, (0, clip_samples - len(c)))
        chunks.append(c)
    return np.stack(chunks)


class MelFrontend(nn.Module):
    def __init__(self):
        super().__init__()
        import torchaudio
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
        self.to_db = torchaudio.transforms.AmplitudeToDB()

    @torch.no_grad()
    def forward(self, wav):               # (B, samples) -> (B, n_mels, T)
        return self.to_db(self.mel(wav))
