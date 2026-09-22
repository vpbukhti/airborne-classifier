from __future__ import annotations

import numpy as np
import librosa

from .constants import (
    CHUNK_SECONDS,
    FIXED_FRAMES,
    HOP_LENGTH,
    HOP_SECONDS,
    MIN_CHUNK_SECONDS,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
)


def load_waveform(path, sr: int = SAMPLE_RATE, offset: float = 0.0, duration: float | None = None) -> np.ndarray:
    waveform, _ = librosa.load(path, sr=sr, mono=True, offset=offset, duration=duration)
    return waveform.astype(np.float32)


def get_duration_seconds(path) -> float:
    return float(librosa.get_duration(path=path))


def chunk_offsets(
    total_seconds: float,
    chunk_seconds: float = CHUNK_SECONDS,
    hop_seconds: float = HOP_SECONDS,
    min_chunk_seconds: float = MIN_CHUNK_SECONDS,
) -> list[tuple[float, float]]:
    """Split a song's duration into (offset, duration) windows.

    A trailing window shorter than min_chunk_seconds is dropped, unless it's the
    only window a (very short) song would otherwise produce.
    """
    if total_seconds <= 0:
        return []
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < total_seconds:
        remaining = total_seconds - start
        duration = min(chunk_seconds, remaining)
        if duration < min_chunk_seconds and windows:
            break
        windows.append((start, duration))
        start += hop_seconds
    return windows


def waveform_to_logmel(
    waveform: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    fixed_frames: int = FIXED_FRAMES,
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = _pad_or_truncate(log_mel, fixed_frames)
    return log_mel.astype(np.float32)


def _pad_or_truncate(spec: np.ndarray, fixed_frames: int) -> np.ndarray:
    n_frames = spec.shape[1]
    if n_frames == fixed_frames:
        return spec
    if n_frames > fixed_frames:
        return spec[:, :fixed_frames]
    pad_width = fixed_frames - n_frames
    return np.pad(spec, ((0, 0), (0, pad_width)), mode="constant", constant_values=spec.min())
