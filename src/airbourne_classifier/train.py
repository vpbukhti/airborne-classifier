from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .constants import (
    CHUNK_SECONDS,
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_FILENAME,
    FIXED_FRAMES,
    HOP_LENGTH,
    HOP_SECONDS,
    LABELS,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
)
from .dataset import AirbourneChunkDataset, build_chunk_index, scan_dataset, split_songs
from .model import SmallAudioCNN


@dataclass
class TrainResult:
    history: list[dict] = field(default_factory=list)
    val_metrics: dict | None = None
    test_metrics: dict | None = None
    model_path: Path | None = None


def _audio_config() -> dict:
    return {
        "sample_rate": SAMPLE_RATE,
        "chunk_seconds": CHUNK_SECONDS,
        "hop_seconds": HOP_SECONDS,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "fixed_frames": FIXED_FRAMES,
        "labels": LABELS,
    }


def train(
    data_dir,
    model_out=None,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-3,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    device: str | None = None,
    num_workers: int = 2,
) -> TrainResult:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_out = Path(model_out) if model_out else DEFAULT_MODEL_DIR / DEFAULT_MODEL_FILENAME
    model_out.parent.mkdir(parents=True, exist_ok=True)

    songs = scan_dataset(data_dir)
    train_songs, val_songs, test_songs = split_songs(songs, val_frac=val_frac, test_frac=test_frac, seed=seed)

    train_loader = DataLoader(
        AirbourneChunkDataset(build_chunk_index(train_songs)),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    val_loader = (
        DataLoader(AirbourneChunkDataset(build_chunk_index(val_songs)), batch_size=batch_size, num_workers=num_workers)
        if val_songs else None
    )
    test_loader = (
        DataLoader(AirbourneChunkDataset(build_chunk_index(test_songs)), batch_size=batch_size, num_workers=num_workers)
        if test_songs else None
    )

    model = SmallAudioCNN(n_classes=len(LABELS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    result = TrainResult(model_path=model_out)
    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        if val_loader:
            val_loss, val_acc = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        else:
            val_loss, val_acc = train_loss, train_acc
        elapsed = time.time() - start
        result.history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "seconds": elapsed,
            }
        )
        print(
            f"epoch {epoch:02d}/{epochs} | train_loss {train_loss:.3f} acc {train_acc:.3f} "
            f"| val_loss {val_loss:.3f} acc {val_acc:.3f} | {elapsed:.1f}s"
        )
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    result.val_metrics = {"accuracy": best_val_acc}
    if test_loader:
        test_loss, test_acc = _run_epoch(model, test_loader, criterion, optimizer, device, train=False)
        result.test_metrics = {"accuracy": test_acc, "loss": test_loss}

    checkpoint = {
        "model_state": model.state_dict(),
        "audio_config": _audio_config(),
        "history": result.history,
    }
    torch.save(checkpoint, model_out)
    print(f"saved checkpoint to {model_out}")
    return result


def _run_epoch(model, loader, criterion, optimizer, device, train: bool) -> tuple[float, float]:
    model.train(mode=train)
    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += inputs.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)
