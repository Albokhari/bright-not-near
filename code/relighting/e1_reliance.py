"""§16 E1 — causal brightness-reliance probe (the spine result).

For each image × model in {DAV2-base, DAV2-§13-cured, DAV2-N4-cured, EndoOmni}, render consistent / adversarial
/ 3 nulls, re-infer, and measure the reliance swing (reliance.py). Aggregate per model with image-clustered
bootstrap CI; Holm. The clean-discordant guard reproduces RESULTS_finetune (DAV2 ~0.71, EndoOmni ~0.85) — this
validates the SCORING/PREPROCESSING path is identical to the §13 finetune path (the relight OPERATOR itself is
validated separately by relight._selftest, which asserts the swing direction).

Three nulls (audit-hardened): null_global (global brightness magnitude); null_local (localized order-preserving
bump → same per-model downsampling as the treatment, controls resolution-dependent bump attenuation);
null_scram (same spatial footprint + gain magnitudes as adversarial, RANDOM point assignment → isolates the
brightness↔depth-order link). Headline reliance = swing − null_scram. The between-model ordering uses a PAIRED
image-clustered bootstrap on per-image reliance differences (NOT CI-overlap), Holm-corrected.

Run (GPU):  python3 e1_reliance.py --models DAV2-base,DAV2-s13,DAV2-n4,EndoOmni
            python3 e1_reliance.py --n 4      (quick pipeline check)
"""
import os, sys, json, argparse
import numpy as np
import cv2

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "validation"))
import pool_inference as PI
import reliance as RZ
import finetune_dav2 as FD
import p1_common as C

P1 = FD.P1
IMG_ROOT = FD.IMG_ROOT
GTD = FD.GTD
FOLDS = FD.FOLDS


def build_adapters(which):
    out = {}
    if "DAV2-base" in which:
        out["DAV2-base"] = PI.DAVAdapter("DAV2-base", fold_heads=None)
    if "DAV2-s13" in which:
        # $DAV2_S13_RUN overrides the ordinal-FT head run (e.g. dav2cd for cluster-disjoint folds)
        out["DAV2-s13"] = PI.DAVAdapter("DAV2-s13", fold_heads=PI.load_dav2_fold_heads(os.environ.get("DAV2_S13_RUN", "dav2")))
    if "DAV2-n4" in which:
        # $DAV2_N4_RUN overrides the cure head run (e.g. dav2conscd for cluster-disjoint folds)
        out["DAV2-n4"] = PI.DAVAdapter("DAV2-n4", fold_heads=PI.load_dav2_fold_heads(os.environ.get("DAV2_N4_RUN", "dav2cons")))
    if "DAV2-deconf" in which:
        out["DAV2-deconf"] = PI.DAVAdapter("DAV2-deconf", fold_heads=PI.load_single_head("deconf_ss_n3000"))
    if "EndoOmni" in which:
        out["EndoOmni"] = PI.EndoOmniAdapter("EndoOmni")
    return out


def item(fn):
    bgr = cv2.imread(os.path.join(IMG_ROOT, fn))
    im = GTD[fn]
    pts = sorted(im["points"], key=lambda p: p["point_id"])
    xy = np.array([[p["x"], p["y"]] for p in pts], float)
    ranks = np.array([p["rank"] for p in pts], float)
    return bgr, xy, ranks, (bgr.shape[0], bgr.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="limit #images (0 = all 307)")
    ap.add_argument("--models", default="DAV2-base,DAV2-s13,EndoOmni")
    ap.add_argument("--out", default="results/counterfactual/E1_reliance.json")
    args = ap.parse_args()

    which = args.models.split(",")
    adapters = build_adapters(which)
    files = sorted(GTD.keys())
    if args.n:
        files = files[: args.n]
    print(f"E1 reliance probe | {len(files)} imgs | models {list(adapters)} | dev {PI.DEV}", flush=True)

    KEYS = ("clean", "clean_disc", "consistent", "adversarial",
            "swing", "null_global", "null_local", "null_scram", "reliance")
    per = {m: {k: [] for k in KEYS} for m in adapters}
    used_files = []
    for n, fn in enumerate(files):
        bgr, xy, ranks, ohw = item(fn)
        if bgr is None:
            continue
        used_files.append(fn)
        fold = PI.fold_of(FOLDS, fn)
        for m, ad in adapters.items():
            f = fold if m == "DAV2-s13" else None
            r = RZ.reliance_for_image(ad, bgr, xy, ranks, ohw, fold=f)
            for k in per[m]:
                per[m][k].append(r[k])
        if (n + 1) % 25 == 0:
            print(f"  {n+1}/{len(files)}", flush=True)

    A = {m: {k: np.array(v, float) for k, v in per[m].items()} for m in adapters}

    def ci(x):
        return list(C.cluster_bootstrap_ci(x))

    # aggregate + image-clustered CI + Holm across models on reliance>0 (within-model)
    res = {}; pvec = []
    for m in adapters:
        d = A[m]
        rng = np.random.default_rng(0); v = d["reliance"][~np.isnan(d["reliance"])]
        bs = v[rng.integers(0, len(v), (10000, len(v)))].mean(1)
        p = float((bs <= 0).mean()); pvec.append(p)
        res[m] = dict(clean_pairwise=float(np.nanmean(d["clean"])),
                      clean_discordant=float(np.nanmean(d["clean_disc"])),
                      consistent=float(np.nanmean(d["consistent"])),
                      adversarial=float(np.nanmean(d["adversarial"])),
                      swing=ci(d["swing"]), null_global=ci(d["null_global"]),
                      null_local=ci(d["null_local"]), null_scram=ci(d["null_scram"]),
                      reliance=ci(d["reliance"]), p_reliance_le0=p)
    for m, pa in zip(adapters, C.holm(pvec)):
        res[m]["p_holm"] = float(pa)

    # S1 — PAIRED between-model image-clustered bootstrap on per-image reliance differences (Holm)
    names = list(adapters)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    paired = {}; ppv = []
    for a, b in pairs:
        diff = A[a]["reliance"] - A[b]["reliance"]            # index-aligned by image (same used_files order)
        dm, dlo, dhi = C.cluster_bootstrap_ci(diff)
        rng = np.random.default_rng(0); dd = diff[~np.isnan(diff)]
        bs = dd[rng.integers(0, len(dd), (10000, len(dd)))].mean(1)
        p2 = 2 * min((bs <= 0).mean(), (bs >= 0).mean())     # two-sided
        paired[f"{a}-vs-{b}"] = {"dReliance": [dm, dlo, dhi], "p": float(p2)}; ppv.append(p2)
    for k, pa in zip(paired, C.holm(ppv)):
        paired[k]["p_holm"] = float(pa)

    out = {"models": res, "paired_reliance": paired,
           "per_image": {m: {k: A[m][k].tolist() for k in KEYS} for m in adapters}, "files": used_files}
    os.makedirs(os.path.join(P1, os.path.dirname(args.out)), exist_ok=True)
    json.dump(out, open(os.path.join(P1, args.out), "w"), indent=1)

    print("\n=== E1 reliance (causal brightness probe; reliance = swing − scramble-null) ===")
    print(f"{'model':<12}{'clean-disc':>11}{'cons':>7}{'adv':>7}{'swing':>8}{'nGlob':>7}{'nLocal':>8}{'nScram':>8}{'reliance[CI]':>22}{'p_holm':>8}")
    for m in adapters:
        r = res[m]
        print(f"{m:<12}{r['clean_discordant']:>11.3f}{r['consistent']:>7.3f}{r['adversarial']:>7.3f}"
              f"{r['swing'][0]:>+8.3f}{r['null_global'][0]:>+7.3f}{r['null_local'][0]:>+8.3f}{r['null_scram'][0]:>+8.3f}"
              f"{r['reliance'][0]:>+11.3f}[{r['reliance'][1]:+.2f},{r['reliance'][2]:+.2f}]{r['p_holm']:>8.3f}")
    print("\n--- paired between-model reliance Δ (image-clustered, Holm) ---")
    for k, pr in paired.items():
        d = pr["dReliance"]
        print(f"  {k:<26} Δrel {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]  p_holm {pr['p_holm']:.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
