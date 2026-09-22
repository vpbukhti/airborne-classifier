from __future__ import annotations

from pathlib import Path

LABELS = ["meaningful", "meaningless", "non_airbourne"]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}

SAMPLE_RATE = 16_000
CHUNK_SECONDS = 10.0
HOP_SECONDS = 10.0  # non-overlapping chunks by default
MIN_CHUNK_SECONDS = 3.0  # drop a trailing chunk shorter than this, unless it's a song's only chunk

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
FIXED_FRAMES = int(CHUNK_SECONDS * SAMPLE_RATE / HOP_LENGTH) + 1  # frames per chunk after pad/truncate

AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac")

DEFAULT_MODEL_FILENAME = "airbourne_classifier.pt"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_MODEL_DIR = REPO_ROOT / "models"
