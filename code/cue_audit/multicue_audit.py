"""Multi-cue spurious-correlation AUDIT of the GT-free ordinal benchmark.

Generalises the single brightness confound into a FRAMEWORK: for a family of appearance cues that
a depth model could exploit as a shortcut, we (1) build a zero-parameter baseline that ranks the 5
points by that cue and score it on the 307-image benchmark (EM / pairwise, image-clustered CI), and
(2) for each learned model, measure a per-cue CONFOUND INDEX = all-pairs accuracy - cue-adversarial
accuracy (pairs where the cue order disagrees with GT). A cue that lets a zero-param baseline score
like SOTA, and on which strong models drop, is a genuine evaluation confound.

Cues (each defined so 'higher = predicted nearer', a-priori physically motivated):
  brightness   BT.601 luminance at point                     (bright = near; the known confound)
  radial       -distance from image centre                   (centre = near; endoscope tube geometry)
  vertical     y / H                                          (bottom = near; natural-image prior)
  sharpness    variance-of-Laplacian in a patch              (in focus = near)
  specular     -distance to nearest specular highlight        (near wet mucosa faces the light)
  contrast     local std/mean in a patch                      (texture visible = near)
  saturation   HSV S at point                                 (redder tissue = near)

Outputs (results/validation/multicue/): cue_baselines.csv, model_confound.csv, per_point.csv, AUDIT.md.
Pure CPU; module env numpy/scipy/cv2/PIL.
"""
import os, sys, json, csv, glob
import numpy as np
import cv2

P1 = "/well/rittscher/users/fxh757/Code/Papers/P1_Ordinal_Endo_Depth_Benchmark__EndoLINA_MICCAI2026"
IMG_ROOT = "/well/rittscher/users/fxh757/Code/Kvasir_Ranking/Merged_PASS/images"
GT = json.load(open(os.path.join(P1, "results/consensus_GT/merged_pass_consensus_gt.json")))
OUT = os.path.join(P1, "results/validation/multicue"); os.makedirs(OUT, exist_ok=True)
CUES = ["brightness", "radial", "vertical", "sharpness", "specular", "contrast", "saturation"]
PATCH = 9  # half-size of the local patch for sharpness/contrast


def cue_nearness_for_image(bgr, pts_xy):
    """Return dict cue -> (5,) nearness scores (higher = predicted nearer)."""
    H, W = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(float)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(float)
    sat = hsv[..., 1]
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    rmax = np.hypot(cx, cy) + 1e-6
    spec = (gray > 0.95 * 255).astype(np.uint8)
    if spec.sum() > 0:
        dist_spec = cv2.distanceTransform(1 - spec, cv2.DIST_L2, 3)
    else:
        dist_spec = np.full_like(gray, np.hypot(H, W), dtype=np.float32)
    out = {c: np.zeros(len(pts_xy)) for c in CUES}
    for k, (x, y) in enumerate(pts_xy):
        xi = int(np.clip(round(x), 0, W - 1)); yi = int(np.clip(round(y), 0, H - 1))
        y0, y1 = max(0, yi - PATCH), min(H, yi + PATCH + 1)
        x0, x1 = max(0, xi - PATCH), min(W, xi + PATCH + 1)
        gp = gray[y0:y1, x0:x1]
        out["brightness"][k] = gray[yi, xi]
        out["radial"][k] = -np.hypot(xi - cx, yi - cy) / rmax
        out["vertical"][k] = yi / max(H - 1, 1)
        out["sharpness"][k] = float(lap[y0:y1, x0:x1].var())
        out["specular"][k] = -float(dist_spec[yi, xi])
        out["contrast"][k] = float(gp.std() / (gp.mean() + 1e-6))
        out["saturation"][k] = sat[yi, xi]
    return out


def ranks_from_nearness(near):
    """Argsort nearness desc -> rank per point (rank 1 = nearest)."""
    order = np.argsort(-np.asarray(near), kind="stable")
    rank = np.empty(len(near), int)
    for pos, idx in enumerate(order):
        rank[idx] = pos + 1
    return rank


def pair_metrics(pred_rank, gt_rank):
    """ExactMatch + all-pairs correctness list (per ordered gt pair i-nearer-j)."""
    n = len(gt_rank)
    em = 1.0 if np.array_equal(pred_rank, gt_rank) else 0.0
    pairs, correct = [], []
    for i in range(n):
        for j in range(n):
            if gt_rank[i] < gt_rank[j]:      # i nearer than j
                pairs.append((i, j))
                correct.append(pred_rank[i] < pred_rank[j])
    return em, np.array(pairs), np.array(correct, float)


def boot_ci(vals, B=10000, seed=0):
    v = np.asarray([x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))], float)
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (B, len(v)))
    bm = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))


def ratio_boot_ci(num_per_img, den_per_img, B=10000, seed=7):
    num = np.asarray(num_per_img, float); den = np.asarray(den_per_img, float)
    keep = den > 0
    num, den = num[keep], den[keep]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(den), (B, len(den)))
    bm = num[idx].sum(1) / den[idx].sum(1)
    return float(num.sum() / den.sum()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))


# ---- load model per-point predicted ranks from scored CSVs (ALL available models) ----
def load_model_pred_ranks():
    """model -> {filename -> {point_id -> pred_rank}}. from_preds_NEW carries pid_k columns; the
    routing/new_models CSVs do NOT, but score_new_model.py writes pred_rank_k in point_id-sorted order, so we
    align positionally to the GT's sorted point_ids. This includes ~20 models (the whole field), not just 4."""
    import glob
    rename = {"Depth Anything V2": "DAV2"}
    keep_main = {"Depth Anything V2", "PPSNet", "2ch", "4ch", "5ch", "7ch", "8ch"}   # distinct models (drop _photo)
    fn2pids = {im["filename"]: [p["point_id"] for p in sorted(im["points"], key=lambda q: q["point_id"])]
               for im in GT["images"]}
    models = {}

    def ingest(path, keep=None, positional=False):
        if not os.path.exists(path):
            return
        with open(path) as f:
            for row in csv.DictReader(f):
                m = row.get("model_name")
                if keep is not None and m not in keep:
                    continue
                m = rename.get(m, m)
                fn = row["filename"]
                try:
                    pr = [int(float(row[f"pred_rank_{k}"])) for k in range(1, 6)]
                except (KeyError, ValueError):
                    continue
                if positional:                                   # pid-less CSV: align to GT sorted point_ids
                    if fn not in fn2pids:
                        continue
                    pids = fn2pids[fn]
                else:
                    try:
                        pids = [int(float(row[f"pid_{k}"])) for k in range(1, 6)]
                    except (KeyError, ValueError):
                        continue
                models.setdefault(m, {}).setdefault(fn, {})
                for pid, r in zip(pids, pr):
                    models[m][fn][pid] = r

    ingest(os.path.join(P1, "results/ordinal_n307_from_preds_NEW/ranking_per_image.csv"), keep_main, positional=False)
    for p in sorted(glob.glob(os.path.join(P1, "results/routing/new_models", "*.csv"))):
        ingest(p, None, positional=True)
    return models


def main():
    # gather per-image geometry: filename -> (point_ids, xy, gt_rank, cue_nearness dict)
    imgs = []
    for im in GT["images"]:
        fn = im["filename"]
        bgr = cv2.imread(os.path.join(IMG_ROOT, fn))
        if bgr is None:
            continue
        pts = sorted(im["points"], key=lambda p: p["point_id"])
        pids = [p["point_id"] for p in pts]
        xy = [(p["x"], p["y"]) for p in pts]
        gt_rank = np.array([p["rank"] for p in pts], int)
        cues = cue_nearness_for_image(bgr, xy)
        imgs.append({"fn": fn, "pids": pids, "gt": gt_rank, "cues": cues})
    print(f"loaded {len(imgs)} images", flush=True)

    # ---- (1) cue baselines ----
    base_rows = []
    per_point_rows = []
    for c in CUES:
        ems, pw_num, pw_den = [], [], []
        for im in imgs:
            cr = ranks_from_nearness(im["cues"][c])
            em, pairs, correct = pair_metrics(cr, im["gt"])
            ems.append(em); pw_num.append(correct.sum()); pw_den.append(len(correct))
            for k, pid in enumerate(im["pids"]):
                if c == "brightness":
                    per_point_rows.append([im["fn"], pid, im["gt"][k]])
        em_m, em_lo, em_hi = boot_ci(ems, seed=1)
        pw_m, pw_lo, pw_hi = ratio_boot_ci(pw_num, pw_den, seed=2)
        base_rows.append([c, em_m, em_lo, em_hi, pw_m, pw_lo, pw_hi])
        print(f"  cue {c:11s} EM {em_m:.3f} [{em_lo:.3f},{em_hi:.3f}]  pairwise {pw_m:.3f} [{pw_lo:.3f},{pw_hi:.3f}]", flush=True)
    with open(os.path.join(OUT, "cue_baselines.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["cue", "exact", "exact_lo", "exact_hi", "pairwise", "pw_lo", "pw_hi"])
        w.writerows(base_rows)

    # ---- (2) per-model per-cue confound index ----
    models = load_model_pred_ranks()
    print(f"models with pred ranks: {list(models.keys())}", flush=True)
    mc_rows = []
    for m, per_fn in models.items():
        for c in CUES:
            all_num, all_den, adv_num, adv_den = [], [], [], []
            for im in imgs:
                fn = im["fn"]
                if fn not in per_fn:
                    continue
                pr_map = per_fn[fn]
                if not all(pid in pr_map for pid in im["pids"]):
                    continue
                mrank = np.array([pr_map[pid] for pid in im["pids"]], int)
                crank = ranks_from_nearness(im["cues"][c])
                gt = im["gt"]
                a_n = a_d = d_n = d_d = 0
                for i in range(5):
                    for j in range(5):
                        if gt[i] < gt[j]:               # i nearer than j (GT)
                            ok = mrank[i] < mrank[j]
                            a_d += 1; a_n += ok
                            if crank[i] > crank[j]:      # cue says i is FARTHER -> adversarial
                                d_d += 1; d_n += ok
                all_num.append(a_n); all_den.append(a_d)
                adv_num.append(d_n); adv_den.append(d_d)
            if sum(all_den) == 0:
                continue
            allp, _, _ = ratio_boot_ci(all_num, all_den, seed=3)
            advp, advlo, advhi = ratio_boot_ci(adv_num, adv_den, seed=4)
            ci = allp - advp
            mc_rows.append([m, c, allp, advp, advlo, advhi, ci, int(sum(adv_den))])
    with open(os.path.join(OUT, "model_confound.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "cue", "all_pairs", "adversarial", "adv_lo", "adv_hi", "confound_index", "n_adv_pairs"])
        w.writerows(mc_rows)

    # ---- markdown summary ----
    with open(os.path.join(OUT, "AUDIT.md"), "w") as f:
        f.write("# Multi-cue confound audit (n=307, image-clustered bootstrap B=1e4)\n\n")
        f.write("## Zero-parameter cue baselines (which cues score like a model?)\n\n")
        f.write("| cue | ExactMatch [95% CI] | pairwise [95% CI] |\n|---|---|---|\n")
        for r in sorted(base_rows, key=lambda z: -z[4]):
            f.write(f"| {r[0]} | {r[1]:.3f} [{r[2]:.3f}, {r[3]:.3f}] | {r[4]:.3f} [{r[5]:.3f}, {r[6]:.3f}] |\n")
        f.write("\n(Benchmark ref: best learned model DAV2 ExactMatch 0.479; brightness floor 0.378.)\n\n")
        f.write("## Per-model per-cue confound index = all-pairs - cue-adversarial (higher = leans on the cue)\n\n")
        by_model = {}
        for r in mc_rows:
            by_model.setdefault(r[0], {})[r[1]] = r
        f.write("| model | " + " | ".join(CUES) + " |\n|" + "---|" * (len(CUES) + 1) + "\n")
        order = sorted(by_model, key=lambda m: -(by_model[m].get("brightness", [0]*7)[6]))  # by brightness confound
        for m in order:
            cells = [f"{by_model[m][c][6]:+.3f}" if c in by_model[m] else "—" for c in CUES]
            f.write(f"| {m} | " + " | ".join(cells) + " |\n")
    print(f"WROTE {OUT} | {len(by_model)} models scored", flush=True)


if __name__ == "__main__":
    main()
