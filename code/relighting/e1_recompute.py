"""§16 E1 — recompute the headline from the saved per-image vectors (no GPU re-run).

Audit fix: the headline causal-reliance metric is the SWING = pairwise(consistent) − pairwise(adversarial)
itself, NOT swing−null_scram. The scramble null with only 2 fixed seeds is a biased estimator of ~0 (it came
out ≈−0.04, signed), which would inflate the metric and DIFFERENTIALLY inflate cross-model gaps. Instead:
  • reliance = swing  (image-clustered CI, Holm)
  • null_global, null_local, null_scram reported as VALIDITY CHECKS (each ≈0 ⇒ non-depth-aligned brightness /
    localized order-preserving / scrambled perturbations do NOT swing the ranking — so the swing is specifically
    caused by the brightness↔depth-ORDER alignment). null_local≈0 is the key control for the audit's S2/S3
    (localized bump, same per-model downsampling, order-preserving).
  • PAIRED between-model image-clustered bootstrap on per-image SWING differences (Holm) for the ordering claim.
Reads results/counterfactual/E1_reliance.json (per_image), writes E1_reliance_final.json.
Run (CPU): python3 e1_recompute.py
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "validation"))
import p1_common as C

RES = os.path.join(C.P1, "results/counterfactual")
J = json.load(open(os.path.join(RES, "E1_reliance.json")))
PI = J["per_image"]
MODELS = list(PI.keys())


def ci(x):
    return list(C.cluster_bootstrap_ci(np.asarray(x, float)))


def paired(a, b, key="swing"):
    diff = np.asarray(PI[a][key], float) - np.asarray(PI[b][key], float)
    dm, lo, hi = C.cluster_bootstrap_ci(diff)
    rng = np.random.default_rng(0); dd = diff[~np.isnan(diff)]
    bs = dd[rng.integers(0, len(dd), (10000, len(dd)))].mean(1)
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return [dm, lo, hi], float(p)


def main():
    res = {}; pvec = []
    for m in MODELS:
        d = PI[m]
        sw = np.asarray(d["swing"], float)
        rng = np.random.default_rng(0); v = sw[~np.isnan(sw)]
        bs = v[rng.integers(0, len(v), (10000, len(v)))].mean(1)
        p = float((bs <= 0).mean()); pvec.append(p)
        res[m] = dict(clean_discordant=float(np.nanmean(d["clean_disc"])),
                      consistent=float(np.nanmean(d["consistent"])), adversarial=float(np.nanmean(d["adversarial"])),
                      reliance_swing=ci(sw), null_global=ci(d["null_global"]),
                      null_local=ci(d["null_local"]), null_scram=ci(d["null_scram"]), p_reliance_le0=p)
    for m, pa in zip(MODELS, C.holm(pvec)):
        res[m]["p_holm"] = float(pa)

    pairs = [(a, b) for i, a in enumerate(MODELS) for b in MODELS[i + 1:]]
    pr = {}; ppv = []
    for a, b in pairs:
        d, p = paired(a, b, "swing")
        pr[f"{a}-vs-{b}"] = {"dSwing": d, "p": p}; ppv.append(p)
    for k, pa in zip(pr, C.holm(ppv)):
        pr[k]["p_holm"] = float(pa)

    json.dump({"models": res, "paired_swing": pr}, open(os.path.join(RES, "E1_reliance_final.json"), "w"), indent=1)
    print("=== E1 causal brightness reliance (headline = SWING; nulls are validity checks, each ≈0) ===")
    print(f"{'model':<12}{'clean-disc':>11}{'cons':>7}{'adv':>7}{'reliance(swing)[CI]':>24}"
          f"{'nGlob':>7}{'nLocal':>8}{'nScram':>8}{'p_holm':>8}")
    for m in MODELS:
        r = res[m]
        print(f"{m:<12}{r['clean_discordant']:>11.3f}{r['consistent']:>7.3f}{r['adversarial']:>7.3f}"
              f"{r['reliance_swing'][0]:>+13.3f}[{r['reliance_swing'][1]:+.2f},{r['reliance_swing'][2]:+.2f}]"
              f"{r['null_global'][0]:>+7.3f}{r['null_local'][0]:>+8.3f}{r['null_scram'][0]:>+8.3f}{r['p_holm']:>8.3f}")
    print("\n--- paired between-model SWING Δ (image-clustered, Holm) — the ordering claim ---")
    for k, v in pr.items():
        d = v["dSwing"]
        sig = "SIG" if v["p_holm"] < 0.05 else "ns"
        print(f"  {k:<26} Δswing {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]  p_holm {v['p_holm']:.3f}  {sig}")
    print("\nwrote E1_reliance_final.json")


if __name__ == "__main__":
    main()
