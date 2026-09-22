# Data layout

Raw audio is never committed to this repo (copyright + size) — `data/raw/` and
`data/test_raw/` are gitignored. Put songs on the Colab runtime (or a local checkout)
at:

```
data/raw/
  meaningful/       # Airbourne songs you've labeled as lyrically meaningful
  meaningless/       # Airbourne songs you've labeled as lyrically meaningless
  non_airbourne/     # the M songs not by Airbourne
```

File names don't matter; any of `.mp3`, `.wav`, `.m4a`, `.flac` are picked up.
`meaningful` + `meaningless` together are your N Airbourne songs; `non_airbourne`
is your M comparison songs.

`airbourne_classifier.train.train()` (and the notebook) automatically splits these
songs (not chunks — whole songs, so nothing leaks between splits) into train/val/test
sets, and prints held-out test accuracy after training. If you'd rather score a fixed,
hand-picked held-out set instead, mirror the same three-folder layout under
`data/test_raw/` and call `airbourne_classifier.evaluate_folder(model, audio_config,
"data/test_raw")` yourself (not wired into the notebook, but available as a library
function) — it returns accuracy plus a confusion matrix.

A useful rule of thumb for a joke project like this: more than ~10 songs per class
lets the small CNN actually generalize a bit rather than memorize; fewer than that
and expect it to mostly memorize specific recordings, which is honestly still funny.
