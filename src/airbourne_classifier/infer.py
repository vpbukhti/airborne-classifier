from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from .audio import chunk_offsets, get_duration_seconds, load_waveform, waveform_to_logmel
from .constants import DEFAULT_MODEL_DIR, DEFAULT_MODEL_FILENAME
from .dataset import scan_dataset
from .model import SmallAudioCNN


def load_model(checkpoint_path=None, device: str | None = None):
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_MODEL_DIR / DEFAULT_MODEL_FILENAME
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    audio_config = checkpoint["audio_config"]
    model = SmallAudioCNN(n_classes=len(audio_config["labels"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, audio_config, device


def predict_song(model, audio_config: dict, path, device: str = "cpu") -> dict:
    total = get_duration_seconds(path)
    windows = chunk_offsets(
        total, chunk_seconds=audio_config["chunk_seconds"], hop_seconds=audio_config["hop_seconds"],
    )
    if not windows:
        raise ValueError(f"could not read any audio from {path}")

    specs = []
    for offset, duration in windows:
        waveform = load_waveform(path, sr=audio_config["sample_rate"], offset=offset, duration=duration)
        specs.append(
            waveform_to_logmel(
                waveform,
                sr=audio_config["sample_rate"],
                n_mels=audio_config["n_mels"],
                n_fft=audio_config["n_fft"],
                hop_length=audio_config["hop_length"],
                fixed_frames=audio_config["fixed_frames"],
            )
        )
    batch = torch.from_numpy(np.stack(specs)).unsqueeze(1).to(device)  # (chunks, 1, n_mels, frames)

    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    mean_probs = probs.mean(axis=0)
    labels = audio_config["labels"]
    label_idx = int(mean_probs.argmax())
    return {
        "label": labels[label_idx],
        "probabilities": {label: float(p) for label, p in zip(labels, mean_probs)},
        "num_chunks": len(windows),
    }


def predict_youtube(model, audio_config: dict, url: str, device: str = "cpu") -> dict:
    """Download a YouTube video's audio to a scratch dir, classify it, then clean up."""
    from .youtube import download_audio

    tmp_dir = tempfile.mkdtemp(prefix="airbourne_yt_")
    try:
        mp3_path = download_audio(url, tmp_dir)
        result = predict_song(model, audio_config, mp3_path, device=device)
        result["source_url"] = url
        return result
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def evaluate_folder(model, audio_config: dict, data_dir, device: str = "cpu") -> dict:
    songs = scan_dataset(data_dir)
    labels = audio_config["labels"]
    confusion = {true: {pred: 0 for pred in labels} for true in labels}
    rows = []
    correct = 0
    for song in songs:
        pred = predict_song(model, audio_config, song.path, device=device)
        confusion[song.label][pred["label"]] += 1
        correct += int(pred["label"] == song.label)
        rows.append({"path": str(song.path), "true": song.label, **pred})
    accuracy = correct / len(songs) if songs else 0.0
    return {"accuracy": accuracy, "confusion_matrix": confusion, "rows": rows}
