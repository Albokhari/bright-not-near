"""§16 E2 — build C3VD ordinal labels (5 ranked points/img) in the merged_pass schema.

Annotator-free: sample 5 spatially-spread, depth-distinct points per C3VD frame and rank them by GT depth
(rank 1 = nearest = smallest depth). Used to fine-tune the EXACT §13 DAV2 head recipe on PHANTOM geometry,
then cross-evaluate on REAL-tissue ordinal — the internally-controlled twin of the EndoSfM3D contrast (same
ViT-L, same loss; only the training DOMAIN differs). A strided subset over the 2889 frames gives video
diversity; points avoid the FOV border (invalid_mask) and the invalid depth=0 region.

Output: results/counterfactual/c3vd_ordinal_gt.json  ({"images":[{filename, points:[{point_id,x,y,rank}]}]})
        results/counterfactual/c3vd_ordinal_split.json ({train:[...], val:[...]})  (video-disjoint by prefix)
Run (CPU): python3 make_c3vd_ordinal.py --stride 5 --val_frac 0.1
"""
import os, sys, glob, json, argparse
import numpy as np
from PIL import Image

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "validation"))
import p1_common as C

C3 = "/well/rittscher/users/fxh757/Datasets/PUBLIC_ENDO/Depth_split_2/test"
OUTDIR = os.path.join(C.P1, "results/counterfactual")


def fov_mask(hw):
    m = np.asarray(Image.open(os.path.join(C3, "invalid_mask.png")).convert("L"))
    m = np.asarray(Image.fromarray(m).resize((hw[1], hw[0]), Image.NEAREST))
    return m > 127        # True = valid


def pick_points(dep, valid, rng, k=5, ncand=400, min_frac=0.06):
    """K points: depth-quantile spread + spatial separation. Returns (xy[K,2] int, ranks[K]). K-generalized:
    quantiles span [.10,.90]; spatial gate loosens as min_frac*sqrt(5/k) so large K can still be placed."""
    ys, xs = np.where(valid & (dep > 0))
    if len(ys) < max(50, 10 * k):
        return None
    H, W = dep.shape
    mind = min_frac * np.sqrt(5.0 / k) * np.hypot(H, W)   # loosen separation for larger K
    ncand = max(ncand, 40 * k)
    d = dep[ys, xs].astype(float)
    qs = np.quantile(d, np.linspace(0.10, 0.90, k))       # K depth quantiles (near -> far)
    chosen = []
    for q in qs:
        cand = rng.choice(len(ys), size=min(ncand, len(ys)), replace=False)
        cand = cand[np.argsort(np.abs(d[cand] - q))]      # closest-depth candidates first
        placed = False
        for ci in cand:
            p = np.array([xs[ci], ys[ci]], float)
            if all(np.hypot(*(p - c)) > mind for c in chosen):
                chosen.append(p); placed = True; break
        if not placed:
            chosen.append(np.array([xs[cand[0]], ys[cand[0]]], float))
    xy = np.array(chosen)
    dv = dep[xy[:, 1].astype(int), xy[:, 0].astype(int)].astype(float)
    if len(np.unique(dv)) < k:
        return None
    ranks = np.argsort(np.argsort(dv)) + 1                 # 1 = nearest (smallest depth)
    return xy.astype(int), ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(OUTDIR, exist_ok=True)

    imgs = sorted(glob.glob(os.path.join(C3, "images", "*")))[:: args.stride]
    deps = {os.path.basename(p): p for p in glob.glob(os.path.join(C3, "depths", "*"))}
    out, mref = [], None
    for ip in imgs:
        fn = os.path.basename(ip)
        if fn not in deps:
            continue
        dep = np.asarray(Image.open(deps[fn])).astype(np.int64)
        if mref is None or mref.shape != dep.shape:
            mref = fov_mask(dep.shape)
        r = pick_points(dep, mref, rng)
        if r is None:
            continue
        xy, ranks = r
        out.append({"filename": fn,
                    "points": [{"point_id": k, "x": int(xy[k, 0]), "y": int(xy[k, 1]), "rank": int(ranks[k])}
                               for k in range(5)]})
    # video-disjoint train/val by filename prefix (e.g. cecum_t1_a)
    vids = sorted({"_".join(im["filename"].split("_")[:-1]) for im in out})
    rng.shuffle(vids)
    nval = max(1, int(len(vids) * args.val_frac))
    valv = set(vids[:nval])
    train = [im["filename"] for im in out if "_".join(im["filename"].split("_")[:-1]) not in valv]
    val = [im["filename"] for im in out if "_".join(im["filename"].split("_")[:-1]) in valv]

    json.dump({"images": out}, open(os.path.join(OUTDIR, "c3vd_ordinal_gt.json"), "w"))
    json.dump({"train": train, "val": val}, open(os.path.join(OUTDIR, "c3vd_ordinal_split.json"), "w"))
    print(f"C3VD ordinal: {len(out)} frames from {len(vids)} videos | train {len(train)} val {len(val)} "
          f"(val videos: {sorted(valv)})")
    # sanity: rank distribution + mean depth-by-rank monotone
    deparr = []
    for im in out[:200]:
        dp = np.asarray(Image.open(deps[im["filename"]])).astype(float)
        deparr.append([dp[p["y"], p["x"]] for p in sorted(im["points"], key=lambda q: q["rank"])])
    md = np.mean(deparr, 0)
    print(f"mean depth by rank (1→5, should INCREASE): {np.round(md / md.max(), 3)}")
    assert np.all(np.diff(md) > 0), "ranks not monotone in depth!"
    print("wrote c3vd_ordinal_gt.json + c3vd_ordinal_split.json — ranks monotone in depth OK")


if __name__ == "__main__":
    main()
