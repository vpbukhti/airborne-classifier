from __future__ import annotations

from .constants import LABELS
from .infer import evaluate_folder, load_model, predict_song, predict_youtube
from .train import train
from .youtube import add_labeled_song, download_audio

__all__ = [
    "LABELS",
    "load_model",
    "predict_song",
    "predict_youtube",
    "evaluate_folder",
    "train",
    "add_labeled_song",
    "download_audio",
]
