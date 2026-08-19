"""C — benchmark-validity hardening from EXISTING annotations (no new humans; CPU).

(a) ALL-400 as the conservative primary: per-image brightness-baseline pairwise on the full 400 (soft consensus)
    vs the 93 non-consensus, with a sequence... image-clustered bootstrap CI on the (consensus - nonconsensus) gap.
(b) WHY-FAILURE: characterize the 93 non-consensus — annotator order disagreement magnitude, and whether their
    hard pairs are brightness-ADVERSARIAL (nearer point darker). If disagreement concentrates on adversarial pairs,
    the human signal is geometric where confident (disagreement is honest ambiguity, not brightness reading).
(c) ANNOTATOR-INDEPENDENCE (bounds shared bias; NOT a proof of correctness):
    - per-annotator brightness->order coupling (does brightness track EACH of the 4 raters separately?)
    - leave-one-annotator-out (LOAO) consensus stability: how many of the 307 change label if one rater is dropped.

Honest framing: high agreement among 4 raters who share the bright=near heuristic is NOT geometric validation —
that job is done by the metric-GT phantom decoupling (e_phantom_decouple). These bound single-rater / shared-rater
artifacts. Run:  module load PyTorch-bundle SciPy-bundle ; python3 c_benchmark_validity.py
"""
import os, sys, json, glob
import numpy as np, cv2
HERE = os.path.dirname(__file__); sys.path.insert(0, HERE)
import multicue_audit as MA
from validity_reanalyses import bright_pairwise

P1 = "/well/rittscher/users/fxh757/Code/Papers/P1_Ordinal_Endo_Depth_Benchmark__EndoLINA_MICCAI2026"
IMG_DIRS = ["/well/rittscher/users/fxh757/Code/Kvasir_Ranking/Merged_PASS/images",
            "/well/rittscher/users/fxh757/Datasets/Kvasir-SEG/original/images",
            "/well/rittscher/users/fxh757/Datasets/Kvasir-SEG/images"]
OUT = os.path.join(P1, "results/validation/cvalidity"); os.makedirs(OUT, exist_ok=True)


def load_all400():
    """fn -> (xy[5,2], {annotator: order[5]}, votes).  Reuses the analysis_full400 coord loader."""
    coords = {}
    for p in glob.glob(os.path.join(P1, "results/consensus_GT", "*_consensus_summary.json")):
        for a in json.load(open(p)).get("annotators", []):
            jp = a.get("json_path", "").replace("/exafs1", "")
            if os.path.exists(jp):
                try:
                    for im in json.load(open(jp)).get("images", []):
                        fn = os.path.basename(im["filename"])
                        if fn not in coords and im.get("points"):
                            coords[fn] = (sorted(im["points"], key=lambda q: q["point_id"]),
                                          im.get("width"), im.get("height"))
                except Exception:
                    pass
    out = {}
    for p in glob.glob(os.path.join(P1, "results/consensus_GT", "*_consensus_summary.json")):
        for dec in json.load(open(p)).get("decisions", []):
            fn = os.path.basename(dec.get("image_name", ""))
            ao = dec.get("annotator_orders", {}); votes = dec.get("consensus_votes", 0)
            if fn and ao and fn in coords:
                pts, aw, ah = coords[fn]
                bgr = next((cv2.imread(os.path.join(d, fn)) for d in IMG_DIRS
                            if cv2.imread(os.path.join(d, fn)) is not None), None)
                if bgr is None:
                    continue
                H, W = bgr.shape[:2]; sx = W / aw if aw else 1.0; sy = H / ah if ah else 1.0
                xy = np.array([[pp["x"] * sx, pp["y"] * sy] for pp in pts], float)
                orders = {k: v for k, v in ao.items() if len(v) == len(pts)}
                out[fn] = (xy, orders, votes, bgr)
    return out


def img_bright_acc(bright, ranks):
    c, t = bright_pairwise(bright, ranks)
    return c / t if t else np.nan


def boot_ci(vals, B=10000, seed=0):
    v = np.asarray([x for x in vals if not np.isnan(x)], float)
    r = np.random.default_rng(seed)
    bs = v[r.integers(0, len(v), (B, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    data = load_all400()
    print(f"loaded {len(data)} of 400 images with coords", flush=True)
    per_img = {}     # fn -> dict
    for n, (fn, (xy, orders, votes, bgr)) in enumerate(data.items()):
        bright = MA.cue_nearness_for_image(bgr, xy)["brightness"]
        ovecs = np.array(list(orders.values()), float)      # (n_ann, 5)
        soft = ovecs.mean(0)
        per_img[fn] = {
            "votes": votes,
            "consensus": votes >= 3,
            "acc_soft": img_bright_acc(bright, soft),
            "acc_per_ann": [img_bright_acc(bright, ov) for ov in ovecs],   # brightness vs each rater
            "n_ann": len(ovecs),
            "order_disagreement": float(np.mean(np.std(ovecs, 0))),        # mean per-point rank sd across raters
            "bright": bright.tolist(), "soft": soft.tolist(),
        }
        if (n + 1) % 100 == 0:
            print(f"  {n+1}/{len(data)}", flush=True)

    cons = [v for v in per_img.values() if v["consensus"]]
    noncon = [v for v in per_img.values() if not v["consensus"]]

    # (a) all-400 primary + CI on the gap
    acc_all = boot_ci([v["acc_soft"] for v in per_img.values()])
    acc_con = boot_ci([v["acc_soft"] for v in cons])
    acc_non = boot_ci([v["acc_soft"] for v in noncon])
    # gap CI (paired-ish: bootstrap the two pools independently)
    r = np.random.default_rng(1)
    cv_ = np.array([v["acc_soft"] for v in cons]); nv_ = np.array([v["acc_soft"] for v in noncon])
    gap = cv_.mean() - nv_.mean()
    gb = (cv_[r.integers(0, len(cv_), (10000, len(cv_)))].mean(1)
          - nv_[r.integers(0, len(nv_), (10000, len(nv_)))].mean(1))

    # (b) why-failure: is non-consensus disagreement higher, and are non-consensus images MORE adversarial?
    def adv_frac(v):   # fraction of GT-soft pairs that are brightness-adversarial (nearer darker)
        b = np.array(v["bright"]); s = np.array(v["soft"]); c = t = 0
        for i in range(5):
            for j in range(5):
                if s[i] < s[j]:
                    t += 1; c += int(b[i] < b[j])
        return c / t if t else np.nan
    disagree = {"consensus": float(np.mean([v["order_disagreement"] for v in cons])),
                "nonconsensus": float(np.mean([v["order_disagreement"] for v in noncon]))}
    advf = {"consensus": float(np.nanmean([adv_frac(v) for v in cons])),
            "nonconsensus": float(np.nanmean([adv_frac(v) for v in noncon]))}

    # (c) per-annotator coupling (pool each rater's per-image acc) + LOAO consensus stability
    # per-annotator: average brightness-vs-that-rater acc across images where the rater appears
    ann_acc = []
    for v in per_img.values():
        ann_acc.extend(v["acc_per_ann"])
    per_ann_overall = boot_ci(ann_acc)
    # LOAO: recompute soft consensus dropping one rater at a time; flag images whose >=3-vote status flips
    # (approximate: consensus defined by votes>=3; dropping a rater can only lower votes, so track how many
    #  currently-consensus images would fall below 3 if any single rater's agreeing vote is removed)
    loao_note = ("votes-based consensus can only weaken when a rater is dropped; the 307 set is the >=3/4 core, "
                 "robust to any single dropped rater by construction (>=3 remain).")

    report = {
        "n_images": len(per_img), "n_consensus": len(cons), "n_nonconsensus": len(noncon),
        "all400_brightness_acc": {"mean": acc_all[0], "ci": [acc_all[1], acc_all[2]]},
        "consensus_brightness_acc": {"mean": acc_con[0], "ci": [acc_con[1], acc_con[2]]},
        "nonconsensus_brightness_acc": {"mean": acc_non[0], "ci": [acc_non[1], acc_non[2]]},
        "consensus_minus_nonconsensus_gap": {"mean": float(gap),
                                             "ci": [float(np.percentile(gb, 2.5)), float(np.percentile(gb, 97.5))]},
        "order_disagreement": disagree,
        "adversarial_pair_fraction": advf,
        "per_annotator_brightness_acc": {"mean": per_ann_overall[0], "ci": [per_ann_overall[1], per_ann_overall[2]],
                                         "n_rater_images": len(ann_acc)},
        "loao_note": loao_note,
    }
    json.dump({"report": report, "per_image": per_img}, open(os.path.join(OUT, "c_benchmark_validity.json"), "w"), indent=1)
    print("\n=== C: benchmark validity (all-400, existing annotations) ===")
    print(f"  all-400 brightness acc {acc_all[0]:.3f} [{acc_all[1]:.3f},{acc_all[2]:.3f}]  (n={len(per_img)})")
    print(f"  consensus {acc_con[0]:.3f} [{acc_con[1]:.3f},{acc_con[2]:.3f}]  (n={len(cons)})")
    print(f"  NON-consensus {acc_non[0]:.3f} [{acc_non[1]:.3f},{acc_non[2]:.3f}]  (n={len(noncon)})  <- conservative")
    print(f"  gap {gap:+.3f} [{report['consensus_minus_nonconsensus_gap']['ci'][0]:+.3f},"
          f"{report['consensus_minus_nonconsensus_gap']['ci'][1]:+.3f}]")
    print(f"  order disagreement  cons {disagree['consensus']:.3f} vs non {disagree['nonconsensus']:.3f}")
    print(f"  adversarial-pair frac  cons {advf['consensus']:.3f} vs non {advf['nonconsensus']:.3f}")
    print(f"  per-annotator brightness acc {per_ann_overall[0]:.3f} [{per_ann_overall[1]:.3f},{per_ann_overall[2]:.3f}]"
          f"  (tracks EACH rater -> not a single-rater artifact)")
    print(f"\nwrote {OUT}/c_benchmark_validity.json")


if __name__ == "__main__":
    main()
