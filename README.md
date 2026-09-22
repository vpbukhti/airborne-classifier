# airbourne-classifier

A joke, mostly: a small neural net that listens to a song and decides whether it's

- an **Airbourne** song with **meaningful** lyrics,
- an **Airbourne** song with **meaningless** lyrics (statistically, most of them), or
- **not Airbourne** at all.

Built to run end-to-end on the free Google Colab **T4** GPU.

## Architecture

Audio-only, timbre-based classification (not a lyrics/semantics model — see the
caveat below):

1. **Preprocess**: each mp3 is resampled to 16kHz mono and split into non-overlapping
   10s chunks (a trailing chunk under 3s is dropped unless it's a song's only chunk).
2. **Features**: each chunk becomes a 64-bin log-mel spectrogram.
3. **Model** (`SmallAudioCNN`, `src/airbourne_classifier/model.py`): 4 conv blocks
   (16→32→64→128 channels, BatchNorm + ReLU + MaxPool) → global average pool →
   2-layer classifier head → 3-way softmax. ~100k parameters — trains in minutes on
   a T4, or even on CPU for a small dataset.
4. **Song-level prediction**: a song's chunks are all scored, and their softmax
   probabilities are averaged to produce one label for the whole song.

Splits are done **by song, not by chunk** (`dataset.split_songs`), so a song never
has some chunks in train and others in test.

**Honest caveat**: "meaningful vs. meaningless" is a property of lyrics, not sound.
This model only hears audio, so it's really learning a per-song acoustic fingerprint
(mix, production, riff) that correlates with however you labeled each track — not
actual lyrical meaning. That's a deliberate simplicity trade-off for this project;
see the PR/commit history if a lyrics-aware hybrid model is ever added.

## Repo layout

```
src/airbourne_classifier/
  constants.py   # audio params, label list, default paths
  audio.py       # mp3 loading, chunking, log-mel spectrograms
  dataset.py     # dataset scanning, song-level splitting, torch Dataset
  model.py       # SmallAudioCNN
  train.py       # training loop -> checkpoint
  infer.py       # single-song prediction, folder evaluation
  storage.py     # git-lfs + Google Drive helpers (see below)
notebooks/
  airbourne_classifier.ipynb   # build a dataset from YouTube links, train, and run inference — same notebook
data/
  README.md      # expected data/raw/{meaningful,meaningless,non_airbourne}/*.mp3 layout
models/
  airbourne_classifier.pt   # the trained checkpoint, once one exists (git-lfs)
tests/           # fast unit tests, no real mp3s or GPU required
```

## Model persistence: git-lfs + Google Drive

The checkpoint (`models/airbourne_classifier.pt`) is tracked with **git-lfs**
(`.gitattributes`) and lives in this GitHub repo — that's the durable, shared copy.
Google Drive is an optional per-user speed cache on top of that. The notebook's flow:

1. On clone/pull, git-lfs pulls whatever checkpoint is currently committed — that's
   the baseline.
2. If you've opted into `USE_DRIVE_CACHE` and a checkpoint exists in your Drive cache
   folder, it **overrides** that baseline (Drive is treated as your freshest working
   copy, since it doesn't require a commit+push cycle to update).
3. Training (only runs if no checkpoint was loaded, or you force it) saves the new
   checkpoint locally, and **automatically** copies it to the Drive cache if
   `USE_DRIVE_CACHE` is on.
4. Publishing a checkpoint back to GitHub via git-lfs is **not part of the notebook** —
   do it deliberately, from a terminal, only when you want this exact checkpoint to
   become the new committed version:

   ```bash
   git lfs track "models/*.pt"   # already set up via .gitattributes
   git add models/airbourne_classifier.pt
   git commit -m "Update trained model checkpoint"
   git push
   ```

   `airbourne_classifier.storage.commit_and_push_model()` wraps the same steps if you'd
   rather script it (e.g. from a Colab cell you add yourself, using a Colab secret named
   `GITHUB_TOKEN` for auth).

## Local dev

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Tests are synthetic (sine waves, empty placeholder files) — no real Airbourne mp3s
or a GPU are needed to run them.

## Using the notebook

Open `notebooks/airbourne_classifier.ipynb` in Google Colab (Runtime → T4 GPU). It:

1. Clones this repo and installs dependencies + git-lfs.
2. Lets you set two things in the Config cell: `USE_DRIVE_CACHE` and Drive/local paths.
3. Lets you build a dataset by pasting YouTube links against a label (form cell) —
   N Airbourne songs split across `meaningful`/`meaningless`, plus M non-Airbourne songs.
4. Trains (or loads an existing checkpoint per the flow above); training prints
   held-out test accuracy automatically.
5. Runs inference on any song by YouTube link.

See `data/README.md` for the underlying folder layout it expects.
