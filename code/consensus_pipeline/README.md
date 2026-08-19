# Benchmark construction: annotation and consensus (paper Section 4)

One line per script (from each file's own header; open the file for the full recipe and paths to adjust).

| script | what it does |
|---|---|
| `auto_annotator.py` | Ring-based Point Sampler for Depth Annotation (Ranker JSON) — Multi-threaded batch |
| `build_missing_ranker_json.py` | Rebuild a missing annotator-block ranker JSON from its raw annotation files. |
| `merge_pass_consensus_to_gt.py` | Merge the PASS-consensus decisions of all owner blocks into the final benchmark ground-truth JSON. |
| `process_rankings.py` | Compute per-annotator orders, agreement, and the >=3/4 exact-order consensus decision per image. |
| `rank_consensus_paper.py` | Aggregate the annotator ranking JSONs, compute per-image consensus, split PASS/FAIL, and export the paper CSV. |
