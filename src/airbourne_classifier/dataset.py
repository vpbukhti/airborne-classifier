from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .audio import chunk_offsets, get_duration_seconds, load_waveform, waveform_to_logmel
from .constants import AUDIO_EXTENSIONS, LABEL_TO_IDX, LABELS


@dataclass(frozen=True)
class Song:
    path: Path
    label: str


def scan_dataset(root_dir) -> list[Song]:
    """Expects root_dir/{label}/*.mp3 for each label in LABELS."""
    root = Path(root_dir)
    songs: list[Song] = []
    for label in LABELS:
        label_dir = root / label
        if not label_dir.is_dir():
            continue
        for path in sorted(label_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                songs.append(Song(path=path, label=label))
    if not songs:
        raise FileNotFoundError(f"No audio files found under {root}. Expected subfolders named {LABELS}.")
    return songs


def split_songs(
    songs: list[Song], val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42
) -> tuple[list[Song], list[Song], list[Song]]:
    """Song-level, stratified-by-label split so a song's chunks never cross splits."""
    by_label: dict[str, list[Song]] = {label: [] for label in LABELS}
    for song in songs:
        by_label[song.label].append(song)

    rng = random.Random(seed)
    train: list[Song] = []
    val: list[Song] = []
    test: list[Song] = []
    for items in by_label.values():
        items = items[:]
        rng.shuffle(items)
        n = len(items)
        n_val = max(1, round(n * val_frac)) if n > 1 else 0
        n_test = max(1, round(n * test_frac)) if n > 2 else 0
        n_val = min(n_val, n)
        n_test = min(n_test, n - n_val)
        test.extend(items[:n_test])
        val.extend(items[n_test : n_test + n_val])
        train.extend(items[n_test + n_val :])
    return train, val, test


@dataclass(frozen=True)
class Chunk:
    path: Path
    label: str
    offset: float
    duration: float


def build_chunk_index(songs: list[Song]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for song in songs:
        total = get_duration_seconds(song.path)
        for offset, duration in chunk_offsets(total):
            chunks.append(Chunk(path=song.path, label=song.label, offset=offset, duration=duration))
    return chunks


class AirbourneChunkDataset(Dataset):
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int):
        chunk = self.chunks[idx]
        waveform = load_waveform(chunk.path, offset=chunk.offset, duration=chunk.duration)
        log_mel = waveform_to_logmel(waveform)
        tensor = torch.from_numpy(log_mel).unsqueeze(0)  # (1, n_mels, frames)
        return tensor, LABEL_TO_IDX[chunk.label]
