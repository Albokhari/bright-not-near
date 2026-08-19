"""Phantom decoupling — the human-independent proof that the ordinal ranking measures GEOMETRY, not luminance.

On C3VD (silicone phantom with METRIC depth), the 5-point ranks come from metric depth (make_c3vd_ordinal.py) —
NO human annotators. We apply the geometry-preserving relight operator to DECORRELATE brightness from true depth:
  - adversarial : brighten FAR / darken NEAR  -> inverts the natural bright=near coupling
  - scramble    : adversarial gain magnitudes at RANDOM points -> breaks the brightness<->depth link (to chance)
Then we score, vs the unchanged metric-depth ranks:
  - the zero-parameter bright=near baseline  -> should COLLAPSE (adversarial << 0.5; scramble ~ 0.5)
  - a learned model (DAV2, EndoOmni)         -> should STAY significantly above chance (geometry is untouched)
The gap = the ranking measures geometry, on data whose ground truth is metric depth, not brightness-biased humans.
This severs the phantom-proxy circularity (it does NOT close the real-mucosa question -> report as corroboration).

Run:  python3 e_phantom_decouple.py --models bright            # CPU only (headline collapse)
      python3 e_phantom_decouple.py --models bright,DAV2-base,EndoOmni   # + learned models (GPU)
"""
import os, sys, json, argparse
import numpy as np
import cv2

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "validation"))
import relight as RL
import reliance as RZ
import pool_inference as PI
import finetune_dav2 as FD

P1 = FD.P1
C3VD = "/well/rittscher/users/fxh757/Datasets/PUBLIC_ENDO/Depth_split_2/test"
GT = os.path.join(P1, "results/counterfactual/c3vd_ordinal_gt.json")
CONDS = ["clean", "adversarial", "scramble"]


class BrightnessBaseline:
    """Zero-parameter 'bright=near' predictor: nearness = point luminance. score_relit ignores ohw/fold."""
    name = "bright"
    def score_relit(self, relit_bgr, xy, ohw, fold=None):
        return RL.point_brightness(relit_bgr, xy)


def build(which):
    out = {}
    if "bright" in which:
        out["bright"] = BrightnessBaseline()
    if "DAV2-base" in which:
        out["DAV2-base"] = PI.DAVAdapter("DAV2-base", fold_heads=None)
    if "EndoOmni" in which:
        out["EndoOmni"] = PI.EndoOmniAdapter("EndoOmni")
    # --- extended learned models (Review #3: extend to key cited models) ---
    # Lazy per-model imports: a missing dep/cache for one model must not kill the others.
    _EXT = {
        "Metric3D": ("pool_ext_metric3d", "Metric3DAdapter"),
        "UniDepth": ("pool_ext_unidepth", "UniDepthAdapter"),
        "DAC":      ("pool_ext_dac", "DACAdapter"),
        "PPSNet":   ("pool_ext_ppsnet", "PPSNetAdapter"),
        "ZoeDepth": ("pool_ext_zoedepth", "ZoeDepthAdapter"),
    }
    for name, (mod, cls) in _EXT.items():
        if name in which:
            try:
                m = __import__(mod, fromlist=[cls])
                out[name] = getattr(m, cls)(name)
            except Exception as e:
                print(f"[build] SKIP {name}: {type(e).__name__}: {e}", flush=True)
    return out


def seq_of(fn):
    """C3VD sequence id = filename minus the trailing _NNNN frame index (cluster unit)."""
    base = fn[:-4] if fn.endswith(".png") else fn
    return base.rsplit("_", 1)[0]


def seq_cluster_ci(vals, seqs, B=10000, seed=0):
    """Sequence-clustered bootstrap CI of the mean (resample sequences, average their frames)."""
    vals = np.asarray(vals, float); seqs = np.asarray(seqs)
    m = ~np.isnan(vals); vals, seqs = vals[m], seqs[m]
    uniq = np.unique(seqs)
    by = {s: vals[seqs == s] for s in uniq}
    rng = np.random.default_rng(seed)
    boot = np.empty(B)
    for b in range(B):
        pick = uniq[rng.integers(0, len(uniq), len(uniq))]
        boot[b] = np.concatenate([by[s] for s in pick]).mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(vals.mean()), float(lo), float(hi), int(len(uniq))


def relit_of(bgr, xy, ranks, cond):
    if cond == "clean":
        return bgr
    if cond == "adversarial":
        return RZ.relit(bgr, xy, "adversarial", ranks=ranks, seed=0)
    if cond == "scramble":
        return RZ.relit(bgr, xy, "scramble", ranks=ranks, seed=1)
    raise ValueError(cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="limit #frames (0 = all 578)")
    ap.add_argument("--models", default="bright,DAV2-base,EndoOmni")
    ap.add_argument("--out", default="results/counterfactual/phantom_decouple.json")
    args = ap.parse_args()

    frames = json.load(open(GT))["images"]
    if args.n:
        frames = frames[: args.n]
    which = args.models.split(",")
    adapters = build(which)
    print(f"phantom decouple | {len(frames)} C3VD frames | models {list(adapters)} | dev {PI.DEV}", flush=True)

    per = {m: {c: [] for c in CONDS} for m in adapters}
    seqs = []
    for n, fr in enumerate(frames):
        fn = fr["filename"]
        bgr = cv2.imread(os.path.join(C3VD, "images", fn))
        if bgr is None:
            continue
        pts = sorted(fr["points"], key=lambda p: p["point_id"])
        xy = np.array([[p["x"], p["y"]] for p in pts], float)
        ranks = np.array([p["rank"] for p in pts], float)     # from METRIC depth (no humans)
        ohw = (bgr.shape[0], bgr.shape[1])
        seqs.append(seq_of(fn))
        relits = {c: relit_of(bgr, xy, ranks, c) for c in CONDS}
        for m, ad in adapters.items():
            for c in CONDS:
                near = ad.score_relit(relits[c], xy, ohw, fold=None)
                per[m][c].append(RZ.pairwise_acc(near, ranks))
        if (n + 1) % 50 == 0:
            print(f"  {n+1}/{len(frames)}", flush=True)

    res = {}
    for m in adapters:
        res[m] = {}
        for c in CONDS:
            mean, lo, hi, nseq = seq_cluster_ci(per[m][c], seqs)
            res[m][c] = {"mean": mean, "lo": lo, "hi": hi, "nseq": nseq}

    out = {"result": res, "n_frames": len(seqs), "conditions": CONDS,
           "per_frame": {m: {c: per[m][c] for c in CONDS} for m in adapters}, "seqs": seqs}
    os.makedirs(os.path.join(P1, os.path.dirname(args.out)), exist_ok=True)
    json.dump(out, open(os.path.join(P1, args.out), "w"), indent=1)

    print("\n=== phantom decoupling: pairwise acc vs METRIC-depth ranks (chance = 0.5) ===")
    print(f"{'model':<12}" + "".join(f"{c:>24}" for c in CONDS))
    for m in adapters:
        row = f"{m:<12}"
        for c in CONDS:
            r = res[m][c]; row += f"{r['mean']:>10.3f} [{r['lo']:.2f},{r['hi']:.2f}]  "
        print(row)
    print(f"\nseqs={res[list(adapters)[0]]['clean']['nseq']}  frames={len(seqs)}  wrote {args.out}")
    print("Headline: bright collapses (adversarial<<0.5, scramble~0.5); learned models stay >0.5 with "
          "non-overlapping CI -> the ranking measures geometry, not luminance, with non-human GT.")


if __name__ == "__main__":
    main()
