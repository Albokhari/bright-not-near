"""Score a monocular depth model on the Bright-not-Near ordinal benchmark.

Reports the three protocol metrics from the paper:
  - pairwise accuracy   fraction of correctly ordered point pairs (10 per image)
  - ExactMatch          fraction of images whose full 5-point order is exactly right
  - discordant accuracy pairwise accuracy on the 391 brightness-discordant pairs,
                        the cue-controlled headline metric

Your model's predictions go in a directory with one file per benchmark image, named by
the Kvasir-SEG filename stem (e.g. cju0qkwl35piu0993l0dewei2.png -> cju0qkwl35piu0993l0dewei2.npy
or .png). Files may be any resolution; points are sampled bilinearly at relative
coordinates. By default larger values mean NEARER (disparity convention); pass
--larger-is farther if your maps are metric depth.

Examples:
  # sanity check: the zero-parameter brightness baseline (needs only the Kvasir images)
  python evaluate.py --baseline brightness --images /path/to/Kvasir-SEG/images
  # expected: pairwise 0.873, ExactMatch 0.365, discordant 0.000

  # your model
  python evaluate.py --pred-dir my_preds/ --larger-is farther --bootstrap 10000

Dependencies: numpy, Pillow (see requirements.txt).
"""
import argparse
import csv
import itertools
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ANN = os.path.join(HERE, "annotations")


def load_benchmark():
    with open(os.path.join(ANN, "consensus_307_points_ranks.json")) as f:
        return json.load(f)["images"]


def load_discordant():
    with open(os.path.join(ANN, "discordant_pairs.csv")) as f:
        rows = list(csv.DictReader(f))
    pairs = {}
    for r in rows:
        pairs.setdefault(r["filename"], []).append(
            (int(r["point_id_a"]), int(r["point_id_b"])))
    return pairs


def read_map(path):
    if path.endswith(".npy"):
        return np.load(path).astype(np.float64)
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:                       # colour PNG: use luminance
        arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return arr.astype(np.float64)


def bilinear(arr, x, y):
    """Sample arr at float (x, y); x, y in arr pixel coordinates."""
    H, W = arr.shape
    x = min(max(x, 0.0), W - 1.0)
    y = min(max(y, 0.0), H - 1.0)
    x0, y0 = int(x), int(y)
    x1, y1 = min(x0 + 1, W - 1), min(y0 + 1, H - 1)
    fx, fy = x - x0, y - y0
    return (arr[y0, x0] * (1 - fx) * (1 - fy) + arr[y0, x1] * fx * (1 - fy)
            + arr[y1, x0] * (1 - fx) * fy + arr[y1, x1] * fx * fy)


def nearness_from_pred(im, pred_dir, ext, larger_is):
    stem = os.path.splitext(im["filename"])[0]
    path = None
    for e in ([ext] if ext else [".npy", ".png"]):
        cand = os.path.join(pred_dir, stem + e)
        if os.path.exists(cand):
            path = cand
            break
    if path is None:
        return None
    arr = read_map(path)
    H, W = arr.shape
    out = {}
    for p in im["points"]:
        x = p["x"] / im["width"] * (W - 1)      # relative coords: any resolution works
        y = p["y"] / im["height"] * (H - 1)
        v = bilinear(arr, x, y)
        out[p["point_id"]] = v if larger_is == "nearer" else -v
    return out


def nearness_brightness(im, images_dir):
    """The paper's zero-parameter baseline: BT.601 luminance at the annotated pixel."""
    path = os.path.join(images_dir, im["filename"])
    if not os.path.exists(path):
        return None
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return {p["point_id"]: gray[int(p["y"]), int(p["x"])] for p in im["points"]}


def score_image(near, points, disc_pairs):
    ranks = {p["point_id"]: p["rank"] for p in points}
    ids = sorted(ranks)
    pw_ok = pw_n = 0
    for a, b in itertools.combinations(ids, 2):
        pw_n += 1
        if near[a] == near[b]:
            pw_ok += 0.5                        # ties (should not occur) score half
        elif (near[a] > near[b]) == (ranks[a] < ranks[b]):
            pw_ok += 1
    pred_order = sorted(ids, key=lambda i: -near[i])
    true_order = sorted(ids, key=lambda i: ranks[i])
    em = float(pred_order == true_order)
    d_ok = d_n = 0
    for a, b in disc_pairs:
        d_n += 1
        if near[a] == near[b]:
            d_ok += 0.5
        elif (near[a] > near[b]) == (ranks[a] < ranks[b]):
            d_ok += 1
    return pw_ok, pw_n, em, d_ok, d_n


def bootstrap_ci(per_img, B, seed=0):
    """Image-clustered percentile bootstrap over (numerator, denominator) rows."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(per_img, dtype=np.float64)
    n = len(arr)
    stats = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        num, den = arr[idx, 0].sum(), arr[idx, 1].sum()
        stats.append(num / den if den else np.nan)
    stats = np.asarray(stats)
    return np.nanpercentile(stats, 2.5), np.nanpercentile(stats, 97.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", help="directory of per-image prediction maps")
    ap.add_argument("--ext", default=None, help="prediction extension (.npy or .png); default: try both")
    ap.add_argument("--larger-is", choices=["nearer", "farther"], default="nearer",
                    help="what a larger prediction value means (default nearer, i.e. disparity)")
    ap.add_argument("--baseline", choices=["brightness"], help="score a built-in baseline instead")
    ap.add_argument("--images", help="Kvasir-SEG image directory (needed for --baseline)")
    ap.add_argument("--bootstrap", type=int, default=0, metavar="B",
                    help="image-clustered bootstrap resamples for 95%% CIs (paper uses 10000)")
    ap.add_argument("--json-out", help="also write the metrics to this JSON file")
    args = ap.parse_args()
    if not args.pred_dir and not args.baseline:
        ap.error("give either --pred-dir or --baseline brightness --images <dir>")
    if args.baseline and not args.images:
        ap.error("--baseline needs --images <Kvasir-SEG image dir>")

    bench = load_benchmark()
    disc = load_discordant()
    pw_rows, em_rows, d_rows = [], [], []
    missing = 0
    for im in bench:
        near = (nearness_brightness(im, args.images) if args.baseline
                else nearness_from_pred(im, args.pred_dir, args.ext, args.larger_is))
        if near is None:
            missing += 1
            continue
        pw_ok, pw_n, em, d_ok, d_n = score_image(near, im["points"],
                                                 disc.get(im["filename"], []))
        pw_rows.append((pw_ok, pw_n))
        em_rows.append((em, 1))
        if d_n:
            d_rows.append((d_ok, d_n))
    if missing:
        print(f"WARNING: no prediction found for {missing}/{len(bench)} images "
              f"(scored on the rest)", file=sys.stderr)
    if not pw_rows:
        sys.exit("no images scored; check --pred-dir / --images and filenames")

    res = {}
    for name, rows in [("pairwise", pw_rows), ("ExactMatch", em_rows), ("discordant", d_rows)]:
        num = sum(r[0] for r in rows)
        den = sum(r[1] for r in rows)
        res[name] = {"value": num / den, "n": int(den)}
        if args.bootstrap:
            lo, hi = bootstrap_ci(rows, args.bootstrap)
            res[name]["ci95"] = [lo, hi]

    src = f"baseline:{args.baseline}" if args.baseline else args.pred_dir
    print(f"\nBright-not-Near benchmark — {len(pw_rows)} images — {src}")
    for name in ["pairwise", "ExactMatch", "discordant"]:
        r = res[name]
        ci = f"  95% CI [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]" if "ci95" in r else ""
        print(f"  {name:<11} {r['value']:.3f}  (n={r['n']}){ci}")
    print("  (discordant is the cue-controlled headline metric; "
          "brightness scores ~0 on it by construction)\n")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(res, f, indent=1)
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()
