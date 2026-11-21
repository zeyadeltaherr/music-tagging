from .cnn import ShortChunkCNN
from .attention_cnn import AttentionCNN
from .ast import PretrainedAST


def build_model(name, n_tags, **kwargs):
    """Model factory used by train.py, build_index.py and the demo."""
    if name == "cnn":
        return ShortChunkCNN(n_tags=n_tags, **kwargs)
    if name == "attn_cnn":
        return AttentionCNN(n_tags=n_tags, **kwargs)
    if name == "ast":
        return PretrainedAST(n_tags=n_tags, **kwargs)
    raise ValueError(f"Unknown model '{name}'. Available: cnn, attn_cnn, ast")
