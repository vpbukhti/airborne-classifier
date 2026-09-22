from __future__ import annotations

import shutil
from pathlib import Path

from .constants import LABELS


def download_audio(url: str, out_dir, filename: str | None = None) -> Path:
    """Download a YouTube video's audio track as an mp3 via yt-dlp.

    Requires ffmpeg on PATH (preinstalled on Colab). Defaults to naming the file
    after the video id, so re-downloading the same URL just overwrites in place
    instead of piling up duplicates.
    """
    import yt_dlp  # imported lazily so the rest of the package works without it installed

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(out_dir / f"{filename or '%(id)s'}.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"},
        ],
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    mp3_path = out_dir / f"{filename or info['id']}.mp3"
    if not mp3_path.is_file():
        raise RuntimeError(f"expected downloaded audio at {mp3_path}, but it wasn't created")
    return mp3_path


def add_labeled_song(url: str, label: str, data_dir, drive_dir=None) -> Path:
    """Download a YouTube video's audio straight into data_dir/<label>/.

    If drive_dir is given, also mirrors the file there so it survives past this
    Colab runtime instead of only living in data/raw (which is gitignored/ephemeral).
    """
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}, got {label!r}")

    label_dir = Path(data_dir) / label
    mp3_path = download_audio(url, label_dir)

    if drive_dir is not None:
        drive_label_dir = Path(drive_dir) / label
        drive_label_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mp3_path, drive_label_dir / mp3_path.name)

    return mp3_path
