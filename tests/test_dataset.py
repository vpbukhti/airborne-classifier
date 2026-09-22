from pathlib import Path

from airbourne_classifier.constants import LABELS
from airbourne_classifier.dataset import Song, scan_dataset, split_songs


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_scan_dataset_reads_label_folders(tmp_path):
    _touch(tmp_path / "meaningful" / "a.mp3")
    _touch(tmp_path / "meaningless" / "b.mp3")
    _touch(tmp_path / "non_airbourne" / "c.mp3")
    _touch(tmp_path / "non_airbourne" / "notes.txt")  # wrong extension, ignored

    songs = scan_dataset(tmp_path)
    assert len(songs) == 3
    assert {s.label for s in songs} == set(LABELS)


def test_split_songs_is_stratified_and_deterministic():
    songs = [Song(path=Path(f"{label}/{i}.mp3"), label=label) for label in LABELS for i in range(6)]

    train_a, val_a, test_a = split_songs(songs, val_frac=0.2, test_frac=0.2, seed=1)
    train_b, val_b, test_b = split_songs(songs, val_frac=0.2, test_frac=0.2, seed=1)

    assert (train_a, val_a, test_a) == (train_b, val_b, test_b)
    assert len(train_a) + len(val_a) + len(test_a) == len(songs)
    for label in LABELS:
        assert any(s.label == label for s in train_a)
