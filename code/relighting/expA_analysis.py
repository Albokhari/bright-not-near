"""Exp A analysis (causal-chain claim 3): does fixed-geometry relight sensitivity S_relight PREDICT
metric-depth degradation ΔE under decorrelation/inversion, ACROSS models?
Reads results/counterfactual/expA/expA_<model>_<dataset>.json (built by expA_run.py).
Cross-model Spearman(S_relight, ΔE_absrel) with model-clustered bootstrap CI, leave-one-model-out (LOMO),
permutation p, and with/without the luminance-only positive control. Writes results/counterfactual/expA/expA_summary.json.
"""
import os, sys, json, glob
import numpy as np
from scipy.stats import spearmanr, pearsonr

P1 = "/well/rittscher/users/fxh757/Code/Papers/P1_Ordinal_Endo_Depth_Benchmark__EndoLINA_MICCAI2026"
EXPA = os.path.join(P1, "results/counterfactual/expA")
REAL = ["dav2", "endoomni", "dac", "unidepth", "metric3d", "ppsnet"]
CONTROLS = ["oracle", "lumonly"]
HARMFUL = ["decorr", "invert"]          # families where brightness stops tracking depth


def load():
    D = {}
    for f in sorted(glob.glob(os.path.join(EXPA, "expA_*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        m, ds = d.get("model"), d.get("dataset")
        if not m:
            continue
        D.setdefault(m, {})[ds] = d.get("per_family", {})
    return D


def _agg(pf_by_ds, families, key):
    """mean over datasets & families of per_family[fam][key]['mean'] (key in {dE_absrel,S_relight})."""
    vals = []
    for ds, pf in pf_by_ds.items():
        for fam in families:
            v = pf.get(fam, {}).get(key, {})
            if isinstance(v, dict) and np.isfinite(v.get("mean", np.nan)):
                vals.append(v["mean"])
    return float(np.mean(vals)) if vals else np.nan


def regression(D, families, models, seed=0, B=10000):
    xs, ys, names = [], [], []
    for m in models:
        if m not in D:
            continue
        x = _agg(D[m], families, "S_relight"); y = _agg(D[m], families, "dE_absrel")
        if np.isfinite(x) and np.isfinite(y):
            xs.append(x); ys.append(y); names.append(m)
    xs, ys = np.array(xs), np.array(ys)
    n = len(xs)
    if n < 3:
        return {"n": n, "note": "too few models"}
    rho = float(spearmanr(xs, ys).correlation)
    pear = float(pearsonr(xs, ys)[0])
    # model-clustered bootstrap CI of Spearman
    rng = np.random.default_rng(seed); boot = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(set(idx)) < 3:
            continue
        boot.append(spearmanr(xs[idx], ys[idx]).correlation)
    boot = np.array([b for b in boot if np.isfinite(b)])
    lo, hi = (np.percentile(boot, [2.5, 97.5]) if boot.size else (np.nan, np.nan))
    # leave-one-model-out
    lomo = [float(spearmanr(np.delete(xs, i), np.delete(ys, i)).correlation) for i in range(n)]
    # permutation p (two-sided): shuffle y
    pr = rng  # reuse
    perm = np.array([spearmanr(xs, pr.permutation(ys)).correlation for _ in range(B)])
    p_perm = float((np.abs(perm) >= abs(rho)).mean())
    return {"n": n, "models": names, "S_relight": [round(v, 4) for v in xs.tolist()],
            "dE_absrel": [round(v, 4) for v in ys.tolist()],
            "spearman": round(rho, 4), "spearman_ci95": [round(float(lo), 4), round(float(hi), 4)],
            "pearson": round(pear, 4), "lomo_min": round(min(lomo), 4), "lomo_max": round(max(lomo), 4),
            "p_perm": round(p_perm, 4)}


def main():
    D = load()
    print(f"loaded models: {sorted(D)}")
    out = {"families_harmful": HARMFUL}
    # per-family per-model table (ΔE_absrel, S_relight) averaged over datasets
    tab = {}
    for m in D:
        tab[m] = {}
        for fam in ["global", "spatial", "physics", "decorr", "invert"]:
            tab[m][fam] = {"dE_absrel": round(_agg(D[m], [fam], "dE_absrel"), 4),
                           "S_relight": round(_agg(D[m], [fam], "S_relight"), 4)}
    out["per_model_family"] = tab
    # HEADLINE regressions on the harmful families (decorr+invert)
    out["regression_realonly"] = regression(D, HARMFUL, REAL)                    # 6 real models
    out["regression_with_lumonly"] = regression(D, HARMFUL, REAL + ["lumonly"])  # + positive control
    out["regression_all"] = regression(D, HARMFUL, REAL + CONTROLS)              # + oracle + lumonly
    # also physics-only + each harmful family alone
    out["regression_physics"] = regression(D, ["physics"], REAL)
    out["regression_decorr"] = regression(D, ["decorr"], REAL)
    out["regression_invert"] = regression(D, ["invert"], REAL)
    json.dump(out, open(os.path.join(EXPA, "expA_summary.json"), "w"), indent=1)

    def show(name, r):
        if r.get("n", 0) < 3:
            print(f"  {name}: {r.get('note','n/a')} (n={r.get('n')})"); return
        print(f"  {name:<26} spearman {r['spearman']:+.3f}  95%CI[{r['spearman_ci95'][0]:+.3f},{r['spearman_ci95'][1]:+.3f}]  "
              f"LOMO[{r['lomo_min']:+.2f},{r['lomo_max']:+.2f}]  p_perm={r['p_perm']:.3f}  n={r['n']}")
    print("\n=== Does S_relight predict ΔE_absrel across models? (decorr+invert) ===")
    show("real models only (6)", out["regression_realonly"])
    show("+ lumonly control", out["regression_with_lumonly"])
    show("+ oracle + lumonly", out["regression_all"])
    print("--- by single family (real models) ---")
    show("physics", out["regression_physics"]); show("decorr", out["regression_decorr"]); show("invert", out["regression_invert"])
    print("\n=== per-model ΔE_absrel under decorr / invert (harmful) + oracle/lumonly controls ===")
    for m in REAL + CONTROLS:
        if m in tab:
            dc, iv = tab[m]["decorr"], tab[m]["invert"]
            print(f"  {m:<10} decorr dE {dc['dE_absrel']:+.3f} S {dc['S_relight']:.3f} | invert dE {iv['dE_absrel']:+.3f} S {iv['S_relight']:.3f}")
    print(f"\nwrote {EXPA}/expA_summary.json")


if __name__ == "__main__":
    main()
