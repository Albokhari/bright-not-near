# Cue audit and validity controls (paper Section 3)

One line per script (from each file's own header; open the file for the full recipe and paths to adjust).

| script | what it does |
|---|---|
| `adversarial_selection.py` | Benchmark v2 — brightness-adversarial point SELECTION on C3VD (annotator-free, dense GT). |
| `annotator_independence.py` | Annotator-independence checks. |
| `c3vd_confound.py` | C3VD cross-dataset confound reproduction (annotator-free). |
| `c_benchmark_validity.py` | Benchmark-validity hardening from EXISTING annotations (no new humans; CPU). |
| `multicue_audit.py` | Multi-cue spurious-correlation AUDIT of the GT-free ordinal benchmark. |
