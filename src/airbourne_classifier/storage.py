from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .constants import DEFAULT_MODEL_DIR, DEFAULT_MODEL_FILENAME


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def mount_drive(mount_point: str = "/content/drive") -> Path:
    if not in_colab():
        raise RuntimeError("Google Drive mounting is only available inside Google Colab.")
    from google.colab import drive  # type: ignore

    drive.mount(mount_point)
    return Path(mount_point)


def restore_from_drive(drive_cache_path, local_path=None) -> bool:
    """Copy a cached checkpoint from Drive into the repo's models/ dir, if present.

    Intended to run after the repo's own git-lfs checkout, so a present Drive cache
    overrides that baseline with whatever was last cached from this notebook.
    """
    local_path = Path(local_path) if local_path else DEFAULT_MODEL_DIR / DEFAULT_MODEL_FILENAME
    drive_cache_path = Path(drive_cache_path)
    if not drive_cache_path.is_file():
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(drive_cache_path, local_path)
    return True


def cache_to_drive(drive_cache_path, local_path=None) -> Path:
    """Copy a trained checkpoint into a Google Drive cache directory."""
    local_path = Path(local_path) if local_path else DEFAULT_MODEL_DIR / DEFAULT_MODEL_FILENAME
    drive_cache_path = Path(drive_cache_path)
    drive_cache_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_path, drive_cache_path)
    return drive_cache_path


def git_lfs_available() -> bool:
    return shutil.which("git-lfs") is not None


def _run(cmd: list[str], cwd) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def commit_and_push_model(
    repo_root,
    model_path=None,
    message: str = "Update trained model checkpoint",
    branch: str = "main",
) -> dict:
    """Track the checkpoint with git-lfs, commit it, and push.

    This is a deliberate, explicit action — call it only when you want to publish the
    current checkpoint as the new committed version. It never runs automatically.
    """
    if not git_lfs_available():
        raise RuntimeError("git-lfs is not installed. Run `apt-get install git-lfs` (Colab) first.")
    repo_root = Path(repo_root)
    model_path = Path(model_path) if model_path else DEFAULT_MODEL_DIR / DEFAULT_MODEL_FILENAME
    rel_path = model_path.relative_to(repo_root)

    _run(["git", "lfs", "install", "--local"], cwd=repo_root)
    _run(["git", "add", ".gitattributes", str(rel_path)], cwd=repo_root)
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo_root)
    if staged.returncode == 0:
        return {"pushed": False, "reason": "no changes to commit"}
    _run(["git", "commit", "-m", message], cwd=repo_root)
    push = _run(["git", "push", "origin", branch], cwd=repo_root)
    return {"pushed": True, "output": push.stdout + push.stderr}
