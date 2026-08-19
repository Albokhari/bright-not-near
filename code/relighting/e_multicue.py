"""PILLAR 5 — Multi-cue counterfactual cue-stability stress test (GPU).

Extends the brightness reliance probe (e1_reliance.py) to TWO further nuisance cues — specular glint and defocus
blur (multicue_ops.py) — and reports, per model, how much its ordinal prediction SWINGS when the cue order is
adversarially flipped, net of controls. Lower swing = more cue-stable.

Per (model, image, cue):
  swing        = pairwise(consistent) − pairwise(adversarial)        [cue-order treatment; >0 ⇒ relies on cue]
  null_matched = pairwise(matched seedA) − pairwise(matched seedB)   [same footprint+magnitudes, RANDOM point
                 assignment → isolates the cue↔depth-order link]  (analogue of E1's scramble null)
  null_away    = pairwise(perturb-off-points) − pairwise(clean)      [same perturbation applied AWAY from the
                 points → must ≈0; catches global side effects]
  reliance     = swing − null_matched

Controls reported once per model: vignette (global radial illumination) swing must ≈0 (a global-shift control),
and null_away above is the localized-away-from-points control. A HUMAN-VERIFICATION subset of perturbed images
is written for author sign-off that geometry is preserved (--dump_verify).

Run (GPU): python3 e_multicue.py --models DAV2-base,DAV2-s13,DAV2-n4,EndoOmni --dump_verify
"""
import os, sys, json, argparse
import numpy as np, cv2

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "validation"))
import pool_inference as PI
import reliance as RZ
import multicue_ops as MC
import e1_reliance as E1
import finetune_dav2 as FD
import p1_common as C

P1 = FD.P1
CUES = ["specular", "blur"]


def cue_reliance(adapter, bgr, xy, ranks, ohw, fold, cue, amp):
    sc = lambda im: adapter.score_relit(im, xy, ohw, fold=fold)
    clean = RZ.pairwise_acc(sc(bgr), ranks)
    a_con = RZ.pairwise_acc(sc(MC.apply(cue, bgr, xy, ranks, "consistent", amp=amp)), ranks)
    a_adv = RZ.pairwise_acc(sc(MC.apply(cue, bgr, xy, ranks, "adversarial", amp=amp)), ranks)
    a_ma = RZ.pairwise_acc(sc(MC.apply(cue, bgr, xy, ranks, "matched", amp=amp, seed=1)), ranks)
    a_mb = RZ.pairwise_acc(sc(MC.apply(cue, bgr, xy, ranks, "matched", amp=amp, seed=2)), ranks)
    a_aw = RZ.pairwise_acc(sc(MC.apply(cue, bgr, xy, ranks, "away", amp=amp, seed=3)), ranks)
    swing = a_con - a_adv
    null_matched = a_ma - a_mb
    null_away = a_aw - clean
    return dict(clean=clean, consistent=a_con, adversarial=a_adv,
                swing=swing, null_matched=null_matched, null_away=null_away,
                reliance=swing - null_matched)


def vignette_swing(adapter, bgr, xy, ranks, ohw, fold):
    sc = lambda im: adapter.score_relit(im, xy, ohw, fold=fold)
    return RZ.pairwise_acc(sc(MC.vignette_apply(bgr, 0.4, False)), ranks) - \
        RZ.pairwise_acc(sc(MC.vignette_apply(bgr, 0.4, True)), ranks)


def dump_verify(files, n=12):
    """Write side-by-side (original | specular-adv | blur-adv | vignette) panels + a manifest for author sign-off."""
    outdir = os.path.join(P1, "results/counterfactual/human_verify"); os.makedirs(outdir, exist_ok=True)
    man = []
    step = max(1, len(files) // n)
    for fn in files[::step][:n]:
        bgr, xy, ranks, _ = E1.item(fn)
        if bgr is None:
            continue
        sp = MC.apply("specular", bgr, xy, ranks, "adversarial")
        bl = MC.apply("blur", bgr, xy, ranks, "adversarial")
        vg = MC.vignette_apply(bgr, 0.4, False)
        for p in (np.asarray(xy, int)):
            for im in (bgr, sp, bl, vg):
                cv2.circle(im, (int(p[0]), int(p[1])), 6, (0, 0, 255), 2)
        panel = np.concatenate([bgr, sp, bl, vg], axis=1)
        cv2.imwrite(os.path.join(outdir, f"verify_{fn}.png"), panel)
        man.append({"filename": fn, "panels": "orig|specular_adv|blur_adv|vignette",
                    "geometry_preserved": "", "notes": ""})
    import csv
    with open(os.path.join(outdir, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "panels", "geometry_preserved", "notes"]); w.writeheader()
        for r in man:
            w.writerow(r)
    print(f"  wrote {len(man)} human-verification panels + manifest.csv to {outdir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--models", default="DAV2-base,DAV2-s13,EndoOmni")
    ap.add_argument("--amp", type=float, default=MC.AMP)
    ap.add_argument("--dump_verify", action="store_true")
    ap.add_argument("--out", default="results/counterfactual/E_multicue.json")
    args = ap.parse_args()

    which = args.models.split(",")
    adapters = E1.build_adapters(which)
    files = sorted(FD.GTD.keys())
    if args.n:
        files = files[: args.n]
    print(f"Pillar5 multi-cue stress | {len(files)} imgs | models {list(adapters)} | cues {CUES} | dev {PI.DEV}", flush=True)
    if args.dump_verify:
        dump_verify(files)

    KEYS = ("clean", "consistent", "adversarial", "swing", "null_matched", "null_away", "reliance")
    per = {m: {c: {k: [] for k in KEYS} for c in CUES} for m in adapters}
    vig = {m: [] for m in adapters}
    used = []
    for n, fn in enumerate(files):
        bgr, xy, ranks, ohw = E1.item(fn)
        if bgr is None:
            continue
        used.append(fn)
        fold = PI.fold_of(FD.FOLDS, fn)
        for m, ad in adapters.items():
            f = fold if m == "DAV2-s13" else None
            for c in CUES:
                r = cue_reliance(ad, bgr, xy, ranks, ohw, f, c, args.amp)
                for k in KEYS:
                    per[m][c][k].append(r[k])
            vig[m].append(vignette_swing(ad, bgr, xy, ranks, ohw, f))
        if (n + 1) % 25 == 0:
            print(f"  {n+1}/{len(files)}", flush=True)

    def ci(x):
        return list(C.cluster_bootstrap_ci(np.array(x, float)))

    res = {}
    for m in adapters:
        res[m] = {"vignette_swing": ci(vig[m]), "cues": {}}
        pvec = []
        for c in CUES:
            d = {k: np.array(per[m][c][k], float) for k in KEYS}
            rng = np.random.default_rng(0); v = d["reliance"][~np.isnan(d["reliance"])]
            bs = v[rng.integers(0, len(v), (10000, len(v)))].mean(1)
            p = float((bs <= 0).mean()); pvec.append(p)
            res[m]["cues"][c] = dict(clean=float(np.nanmean(d["clean"])),
                                     consistent=float(np.nanmean(d["consistent"])),
                                     adversarial=float(np.nanmean(d["adversarial"])),
                                     swing=ci(d["swing"]), null_matched=ci(d["null_matched"]),
                                     null_away=ci(d["null_away"]), reliance=ci(d["reliance"]),
                                     p_reliance_le0=p)
        for c, pa in zip(CUES, C.holm(pvec)):
            res[m]["cues"][c]["p_holm"] = float(pa)

    out = {"models": res, "cues": CUES, "files": used,
           "per_image": {m: {c: {k: [float(x) for x in per[m][c][k]] for k in KEYS} for c in CUES} for m in adapters}}
    os.makedirs(os.path.join(P1, os.path.dirname(args.out)), exist_ok=True)
    json.dump(out, open(os.path.join(P1, args.out), "w"), indent=1)

    print("\n=== Pillar 5 multi-cue cue-stability (reliance = swing − matched-null; lower = more stable) ===")
    print(f"{'model':<12}{'cue':<10}{'clean':>7}{'cons':>7}{'adv':>7}{'swing':>8}{'nMatch':>8}{'nAway':>8}{'reliance[CI]':>22}{'p_holm':>8}")
    for m in adapters:
        for c in CUES:
            r = res[m]["cues"][c]
            print(f"{m:<12}{c:<10}{r['clean']:>7.3f}{r['consistent']:>7.3f}{r['adversarial']:>7.3f}"
                  f"{r['swing'][0]:>+8.3f}{r['null_matched'][0]:>+8.3f}{r['null_away'][0]:>+8.3f}"
                  f"{r['reliance'][0]:>+11.3f}[{r['reliance'][1]:+.2f},{r['reliance'][2]:+.2f}]{r['p_holm']:>8.3f}")
        print(f"{m:<12}{'vignette(ctrl)':<10}{'':>21}{res[m]['vignette_swing'][0]:>+8.3f}"
              f"[{res[m]['vignette_swing'][1]:+.2f},{res[m]['vignette_swing'][2]:+.2f}]  (must ≈0)")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
