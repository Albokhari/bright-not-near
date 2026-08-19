# Relighting operators, reliance probe, and Exp A (paper Section 3)

One line per script (from each file's own header; open the file for the full recipe and paths to adjust).

| script | what it does |
|---|---|
| `cure_heldout_operator.py` | Does the relight-consistency CURE generalize to an INDEPENDENT relight operator? |
| `dense_relight_families.py` | Exp A: dense, GT-depth-keyed, geometry-preserving relight families (unit = image/scene). |
| `e1_recompute.py` | Recompute the headline from the saved per-image vectors (no GPU re-run). |
| `e1_reliance.py` | Causal brightness-reliance probe (the spine result). |
| `e2_crossdomain.py` | E2 — controlled C3VD-vs-real fine-tune (the rigor centerpiece for finding #1). |
| `e3_deconfound_simple.py` | E3-simple (training-free, CPU) — test-time de-confounding by brightness subtraction. |
| `e_multicue.py` | PILLAR 5 — Multi-cue counterfactual cue-stability stress test (GPU). |
| `e_phantom_decouple.py` | Phantom decoupling — the human-independent proof that the ordinal ranking measures GEOMETRY, not luminance. |
| `expA_analysis.py` | Exp A analysis (causal-chain claim 3): does fixed-geometry relight sensitivity S_relight PREDICT |
| `expA_controls.py` | Exp A controls: two reference "models" that anchor the S_relight vs dE relationship. |
| `expA_dense_inference.py` | Exp A :: expA_dense_inference.py — DensePredictor: one .predict(model_name, bgr) -> DENSE depth/disparity |
| `expA_metrics.py` | Exp A metrics: per-image scale+shift alignment + dense metric-depth errors. |
| `expA_run.py` | Exp A main loop — counterfactual relight sensitivity vs. dense metric-depth error. |
| `make_c3vd_ordinal.py` | E2 — build C3VD ordinal labels (5 ranked points/img) in the merged_pass schema. |
| `make_expA_figure.py` | Exp A figure: (a) per-family metric degradation ΔE (the shortcut isolation: brightness change that PRESERVES |
| `make_expA_sample_data.py` | Render qualitative Exp A samples: for a few frames and models, save the relit RGB and the model's |
| `make_expA_samples_fig.py` | Compose the Exp A qualitative-samples figures from rendered panels (make_expA_sample_data.py). |
| `make_spine_figure.py` | Spine figure — the one unifying result: causal brightness reliance (E1 swing) across models/cures. |
| `multicue_ops.py` | PILLAR 5 (multi-cue) — counterfactual operators BEYOND brightness: specular glint + defocus blur, plus a |
| `n1_adapter.py` | N1 — universal de-biasing adapter (super-novel, CPU). Does ONE reusable corrector de-bias ANY model? |
| `pool_ext_dac.py` | Ext pool_ext_dac.py — DepthAnything-AC → score_relit adapter for the metric-GT decorrelation |
| `pool_ext_metric3d.py` | Ext pool_ext_metric3d.py — Metric3D → score_relit adapter for the counterfactual-relight probe. |
| `pool_ext_ppsnet.py` | Pool_ext_ppsnet.py — PPSNet (ECCV 2024, Paruchuri et al.) score_relit adapter for the |
| `pool_ext_unidepth.py` | Ext UniDepthAdapter — score_relit adapter for the metric-GT decorrelation probe (e_phantom_decouple.py). |
| `pool_ext_zoedepth.py` | Ext pool_ext_zoedepth.py — ZoeDepth score_relit adapter for the metric-GT decorrelation probe. |
| `pool_inference.py` | Pool_inference.py — model → score_relit adapters for the counterfactual-relight probe (E1/N4/E3-adv). |
| `reliance.py` | Reliance.py — the spine metric: a per-model CAUSAL brightness-reliance 'swing'. |
| `relight.py` | Shared engine — per-point counterfactual relighting operator. |
| `relight_sideeffects.py` | Earn -- relight SIDE-EFFECT audit + MATCHED controls + PHYSICS arm (CPU-only). |
| `train_lumonly.py` | Train the Exp A `lumonly` positive control: a tiny grayscale-input CNN (TinyLumNet, defined in |
