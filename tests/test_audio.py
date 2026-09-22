import numpy as np
import soundfile as sf

from airbourne_classifier.audio import chunk_offsets, load_waveform, waveform_to_logmel
from airbourne_classifier.constants import FIXED_FRAMES, N_MELS, SAMPLE_RATE


def test_chunk_offsets_splits_long_signal():
    windows = chunk_offsets(total_seconds=25.0, chunk_seconds=10.0, hop_seconds=10.0, min_chunk_seconds=3.0)
    assert windows == [(0.0, 10.0), (10.0, 10.0), (20.0, 5.0)]


def test_chunk_offsets_keeps_short_only_chunk():
    windows = chunk_offsets(total_seconds=1.5, chunk_seconds=10.0, hop_seconds=10.0, min_chunk_seconds=3.0)
    assert windows == [(0.0, 1.5)]


def test_chunk_offsets_drops_short_trailing_chunk():
    windows = chunk_offsets(total_seconds=11.0, chunk_seconds=10.0, hop_seconds=10.0, min_chunk_seconds=3.0)
    assert windows == [(0.0, 10.0)]


def test_waveform_to_logmel_shape():
    waveform = np.sin(2 * np.pi * 440 * np.arange(SAMPLE_RATE * 10) / SAMPLE_RATE).astype(np.float32)
    log_mel = waveform_to_logmel(waveform)
    assert log_mel.shape == (N_MELS, FIXED_FRAMES)


def test_load_waveform_reads_correct_length(tmp_path):
    sr = SAMPLE_RATE
    signal = np.sin(2 * np.pi * 220 * np.arange(sr * 2) / sr).astype(np.float32)
    path = tmp_path / "tone.wav"
    sf.write(path, signal, sr)

    loaded = load_waveform(path, sr=sr)
    assert abs(len(loaded) - sr * 2) < sr * 0.05
