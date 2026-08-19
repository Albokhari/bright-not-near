# Code

Research code, released as used for the paper. Scripts reference cluster paths near the
top of each file; point them at your own copies of the annotations (`../annotations/`),
folds (`../folds/`), Kvasir-SEG images, and the C3VD / SimCol datasets.

| Directory | Paper section | Contents |
|---|---|---|
| `cue_audit/` | Section 3 (multi-cue audit, discordant re-ranking, validity controls) | `multicue_audit.py` (the seven a-priori cues and confound indices), `adversarial_selection.py` (brightness-discordant split scoring), `annotator_independence.py`, `c3vd_confound.py` (annotator-free reproduction on metric GT), `c_benchmark_validity.py` (de-vignetting / albedo / non-consensus controls) |
| `relighting/` | Section 3 (interventional probe, Exp A) | `relight.py` and `multicue_ops.py` (the per-point relighting operator), `dense_relight_families.py` (global / spatial / physics / decorrelate / invert), `reliance.py`, `e1_*`/`e2_*`/`e3_*` (reliance and decorrelation analyses), `expA_*.py` (fixed-geometry relighting), `train_lumonly.py` (luminance-only control) |
| `evaluation/` | Sections 3-4 (ordinal scoring protocol) | `bench_depth_ranking_kvasirseg.py` and `..._from_preds.py`: pairwise / ExactMatch / brightness-discordant scoring of any depth model on the benchmark, with scene-clustered bootstrap and Holm correction |
| `finetune/` | Section 5 (mitigation) | `finetune_dav2.py` (head-only ordinal fine-tune), `finetune_dav2_consistency.py` (+ relight-consistency), `ranking_loss.py` (DIW loss with discordant up-weighting), `make_splits.py`, `eval_head.py`, `eval_stats.py` |
| `consensus_pipeline/` | Section 4 (benchmark construction) | annotation collection, cross-block merging, and the >=3/4 exact-order consensus filter |

`example_loader.py` in this directory shows how to join the released annotations with the
Kvasir-SEG images and iterate the discordant pairs.
