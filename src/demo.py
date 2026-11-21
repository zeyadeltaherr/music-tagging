import json
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi

from src.datasets import build_datasets
from src.models import build_model
from src.similarity import SimilarityIndex


# =========================
# Configuration
# =========================

CHECKPOINT = "checkpoints/ast_official_fbank_seed0.pt"
INDEX_PATH = "checkpoints/ast_similarity_index.npz"

SAMPLE_RATE = 16000
WAV_SAMPLES = int(28.8 * SAMPLE_RATE)
MAX_LENGTH = 1024

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Load Model
# =========================

print("Loading AST model...")

checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
cfg = checkpoint["cfg"]

datasets = build_datasets(cfg["data"])
TAG_NAMES = datasets["tag_names"]

model = build_model(
    cfg["model"]["name"],
    n_tags=checkpoint["n_tags"],
    **cfg["model"].get("params", {})
).to(DEVICE)

model.load_state_dict(checkpoint["model_state"])
model.eval()

print(f"Model loaded on {DEVICE}")


# =========================
# Load Similarity Index
# =========================

print("Loading similarity index...")

index = SimilarityIndex.load(INDEX_PATH)

root = Path(cfg["data"]["root"])
meta = json.loads(
    (root / "ast_fbank" / "meta.json").read_text(encoding="utf-8")
)

metadata = {}

for clip_id, name, song in zip(
    meta["clip_ids"],
    meta["names"],
    meta["songs"]
):
    metadata[str(clip_id)] = {
        "name": name,
        "song": song
    }

print(f"Index loaded: {len(index.emb)} tracks")


# =========================
# Audio Preprocessing
# =========================

def extract_features(audio_path):

    wav, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(wav) < WAV_SAMPLES:
        wav = np.pad(
            wav,
            (0, WAV_SAMPLES - len(wav))
        )

    wav = wav[:WAV_SAMPLES].astype(np.float32)

    waveform = torch.from_numpy(wav).unsqueeze(0)

    fbank = kaldi.fbank(
        waveform,
        sample_frequency=SAMPLE_RATE,
        window_type="hanning",
        num_mel_bins=128,
    )

    if fbank.shape[0] < MAX_LENGTH:
        padding = torch.zeros(
            MAX_LENGTH - fbank.shape[0],
            128
        )
        fbank = torch.cat([fbank, padding], dim=0)
    else:
        fbank = fbank[:MAX_LENGTH]

    return fbank.unsqueeze(0).to(DEVICE)


# =========================
# Prediction
# =========================

def analyze_music(audio_path):

    if audio_path is None:
        raise gr.Error("Please upload an audio file.")

    try:
        features = extract_features(audio_path)

        with torch.no_grad():

            embedding = model.embed(features)

            logits = model.classify(embedding)

            probabilities = torch.sigmoid(logits)[0]

        probabilities = probabilities.cpu().numpy()

        # Top 10 tags
        top_indices = np.argsort(probabilities)[::-1][:10]

        tags = {
            str(TAG_NAMES[i]): float(probabilities[i])
            for i in top_indices
        }

        # Similarity search
        query_embedding = embedding.cpu().numpy()

        indices, scores = index.search(
            query_embedding,
            k=5
        )

        results = []

        for idx, score in zip(indices[0], scores[0]):

            clip_id = str(index.ids[idx])

            info = metadata.get(clip_id, {})

            results.append([
                clip_id,
                info.get("name", "Unknown"),
                info.get("song", "Unknown"),
                round(float(score), 4)
            ])

        return tags, results

    except Exception as e:
        raise gr.Error(f"Analysis failed: {str(e)}")


# =========================
# Gradio Interface
# =========================

with gr.Blocks(title="AI Music Intelligence") as demo:

    gr.Markdown(
        """
        # 🎵 AI Music Intelligence

        ### Music Auto-Tagging & Similarity Search

        Upload an audio file to discover its musical characteristics
        and find similar tracks using a pretrained Audio Spectrogram Transformer.
        """
    )

    audio = gr.Audio(
        sources=["upload"],
        type="filepath",
        label="Upload Music (MP3 / WAV)"
    )

    analyze_btn = gr.Button(
        "Analyze Music",
        variant="primary"
    )

    gr.Markdown("## 🎼 Predicted Music Tags")

    tags_output = gr.Label(
        num_top_classes=10,
        label="Top 10 Tags"
    )

    gr.Markdown("## 🔍 Similar Music Tracks")

    similar_output = gr.Dataframe(
        headers=[
            "Clip ID",
            "Track Name",
            "Song",
            "Cosine Similarity"
        ],
        label="Top 5 Similar Tracks",
        interactive=False
    )

    analyze_btn.click(
        fn=analyze_music,
        inputs=audio,
        outputs=[tags_output, similar_output]
    )

    gr.Markdown(
        """
        ---
        **Model:** AST Official Fbank

        **Dataset:** MagnaTagATune

        **Retrieval:** Cosine Similarity

        **Hardware:** NVIDIA RTX 4050
        """
    )


if __name__ == "__main__":

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )