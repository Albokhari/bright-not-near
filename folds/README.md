# Fold definitions

- `folds.json` — the image-level 5-fold split used for all fine-tuning and 5-fold
  evaluation numbers in the paper.
- `folds_clusterdisjoint.json` — the robustness variant in which near-duplicate frames
  (ORB-matched clusters) are kept within a single fold.

Both map fold indices to lists of Kvasir-SEG filenames.
