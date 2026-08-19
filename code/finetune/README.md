# Mitigation: head-only fine-tuning (paper Section 5)

One line per script (from each file's own header; open the file for the full recipe and paths to adjust).

| script | what it does |
|---|---|
| `eval_head.py` | Fallback: evaluate a saved head_best.pt on the 307 benchmark (base vs ft) and write summary.json. |
| `eval_stats.py` | Decision-gate statistics for the fine-tuning pilot (and full CV). |
| `finetune_dav2.py` | DAV2 head-only fine-tune with DIW ordinal ranking loss (pilot). |
| `finetune_dav2_consistency.py` | Brightness-CONSISTENCY fine-tune (the relight closes the loop, super-novel cure). |
| `make_splits.py` | Fold image-disjoint split of the 307 consensus images (rotating train/val/test CV). |
| `ranking_loss.py` | DIW/WHDR-style pairwise ranking loss for sparse ordinal depth supervision. |
