"""§16 E3-simple (training-free, CPU) — test-time de-confounding by brightness subtraction.

Per image, de-confound nearness as  s(λ) = z(near) − λ·z(brightness)  and re-rank (largest s = nearest).
λ is a single de-confounding STRENGTH tuned on held-out folds (folds.json); λ=0 reproduces the original.

Two operating points, because the metric matters:
  • λ tuned for ExactMatch (the full 5-point order) → held-out λ=0 for every model: subtracting brightness
    never improves the net ranking, because near genuinely IS brighter (removing it removes real geometry).
  • λ tuned for Discordant accuracy (the brightness-ADVERSARIAL pairs — near point darker than far) → a large
    apparent "cure", biggest for the most-confounded models. The DISCRIMINATOR (the §15 permutation-null lesson)
    is what that same λ does to ExactMatch and to the CONCORDANT pairs: a genuine cure lifts discordant WITHOUT
    sinking concordant/exact; a mechanical "boost-dark" re-allocation just trades concordant for discordant and
    nets ~0 on ExactMatch. Controls: shuffle-brightness null + image-clustered bootstrap CI.
Run: python3 e3_deconfound_simple.py
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "validation"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "routing"))
import p1_common as C

P1 = C.P1
NEW = os.path.join(P1, "results/routing/new_models")
PT = [1, 2, 3, 4, 5]
MODELS = ["Depth Anything V2", "PPSNet", "EndoOmni", "SHADeS", "Marigold"]
NEWMODELS = ["DepthPro", "DUSt3R", "EndoDAC", "Metric3D", "MiDaS", "MoGe2", "UniDepth",
             "ZoeDepth", "DepthAnythingAC", "EndoSfM3D_C3VD", "ColonAdapter", "SSL4GIE_HK", "SSL4GIE_IN"]
DISP = {"Depth Anything V2": "DAV2"}
LAMS = np.round(np.arange(0.0, 2.01, 0.1), 2)
FOLDS = json.load(open(os.path.join(P1, "results/finetune/folds.json")))


def _z(x):
    x = np.asarray(x, float); s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else x - x.mean()


def load_model_rows(df_wide, bval, model):
    """Per filename: (nearness[5] oriented larger=nearer, gt_rank[5], brightness[5])."""
    if model in MODELS:
        sub = df_wide[df_wide.model_name == model]
        rows = {r.filename: (np.array([r[f"pred_val_{k}"] for k in PT], float),
                             np.array([r[f"pred_rank_{k}"] for k in PT], float),
                             np.array([r[f"gt_rank_{k}"] for k in PT], float)) for _, r in sub.iterrows()}
    else:
        p = os.path.join(NEW, f"{model}.csv")
        if not os.path.exists(p):
            return None
        nm = pd.read_csv(p)
        rows = {r.filename: (np.array([r[f"pred_val_{k}"] for k in PT], float),
                             np.array([r[f"pred_rank_{k}"] for k in PT], float),
                             np.array([r[f"gt_rank_{k}"] for k in PT], float)) for _, r in nm.iterrows()}
    allv = np.concatenate([v for v, _, _ in rows.values()])
    allr = np.concatenate([r for _, r, _ in rows.values()])
    sgn = 1.0 if spearmanr(allv, allr).correlation < 0 else -1.0
    out = {}
    for f, (v, pr, gt) in rows.items():
        if f in bval:
            out[f] = (sgn * v, gt, np.asarray(bval[f], float))
    return out


def build_cache(rows):
    """Per file precompute znear, zbright, gt nearest-first order, and adversarial/concordant pair indices.
    A pair (i,j) with gt[i]<gt[j] (i is nearer): discordant if bright[i]<bright[j], concordant if bright[i]>bright[j].
    'Correct' on a pair = predicted nearness s[i] > s[j]."""
    cache = {}
    for f, (near, gt, br) in rows.items():
        gt_order = tuple(np.argsort(gt))                      # nearest-first GT order
        di, dj, ci, cj = [], [], [], []
        for i in range(5):
            for j in range(5):
                if gt[i] < gt[j]:
                    if br[i] < br[j]:
                        di.append(i); dj.append(j)
                    elif br[i] > br[j]:
                        ci.append(i); cj.append(j)
        cache[f] = dict(zn=_z(near), zb=_z(br), gt_order=gt_order,
                        di=np.array(di, int), dj=np.array(dj, int),
                        ci=np.array(ci, int), cj=np.array(cj, int))
    return cache


def score_all(c, lam):
    """Return (exact, disc, conc) for one image cache at strength λ (vectorized)."""
    s = c["zn"] - lam * c["zb"]
    ex = float(tuple(np.argsort(-s)) == c["gt_order"])
    dc = float(np.mean(s[c["di"]] > s[c["dj"]])) if c["di"].size else np.nan
    cc = float(np.mean(s[c["ci"]] > s[c["cj"]])) if c["ci"].size else np.nan
    return ex, dc, cc


MIDX = {"exact": 0, "disc": 1, "conc": 2}


def oof_scores(cache, tune_metric, shuffle=False, seed=1):
    """Held-out: per fold tune λ on the other folds (max tune_metric), return per-image (exact,disc,conc) at that λ."""
    rng = np.random.default_rng(seed)
    cc = cache
    if shuffle:                                               # permute brightness within each image
        cc = {}
        for f, c in cache.items():
            d = dict(c); d["zb"] = rng.permutation(c["zb"]); cc[f] = d
    fidx = {f: k for k, rnd in enumerate(FOLDS["rounds"]) for f in rnd["test"] if f in cc}
    mi = MIDX[tune_metric]
    out = {}; lams = []
    # precompute score table [file][lam] -> (ex,dc,cc) once
    tab = {f: [score_all(cc[f], lam) for lam in LAMS] for f in cc}
    for k, rnd in enumerate(FOLDS["rounds"]):
        test = [f for f in rnd["test"] if f in cc]
        tune = [f for f in cc if fidx.get(f) != k]
        best_l, best = 0, -1
        for li in range(len(LAMS)):
            v = np.nanmean([tab[f][li][mi] for f in tune])
            if v > best:
                best, best_l = v, li
        lams.append(LAMS[best_l])
        for f in test:
            out[f] = tab[f][best_l]
    return out, lams


def boot(delta, B=10000):
    delta = delta[~np.isnan(delta)]
    rng = np.random.default_rng(0); n = len(delta); idx = rng.integers(0, n, (B, n))
    d = delta[idx].mean(1)
    return float(delta.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    dfw = C.load_wide(); bval, _ = C.brightness_lookup(dfw)
    print("Test-time de-confounding  s(λ)=z(near) − λ·z(bright), λ held-out tuned (folds.json).")
    print("EM-tuned λ = best λ for ExactMatch (net ranking). DISC-tuned λ = best λ for the adversarial pairs;")
    print("the trade-off it induces (Δexact / Δconc / Δdisc) is the discriminator. GENUINE = disc↑ beating the")
    print("shuffle-null WITHOUT hurting exact; re-alloc = disc↑ bought by conc↓ & exact flat.\n")
    print(f"{'model':<14}{'EM-tuned ΔEM':>16}|{' DISC-tuned:  Δexact':>22}{'Δconc':>9}{'Δdisc[CI]':>20}{'  λ':>5}  verdict")
    res = {}
    for m in MODELS + NEWMODELS:
        rows = load_model_rows(dfw, bval, m)
        if not rows:
            continue
        cache = build_cache(rows)
        files = sorted(cache)
        base = np.array([score_all(cache[f], 0.0) for f in files])      # (N,3) orig exact/disc/conc
        # op-point 1: tune λ on ExactMatch
        oem, _ = oof_scores(cache, "exact")
        em1 = np.array([oem[f] for f in files])
        d_em = boot(em1[:, 0] - base[:, 0])
        # op-point 2: tune λ on Discordant, then read all three
        odc, lams = oof_scores(cache, "disc")
        dd = np.array([odc[f] for f in files])
        d_ex = boot(dd[:, 0] - base[:, 0]); d_cc = boot(dd[:, 2] - base[:, 2]); d_dc = boot(dd[:, 1] - base[:, 1])
        sdc, _ = oof_scores(cache, "disc", shuffle=True)
        ss = np.array([sdc[f] for f in files])
        s_dc = boot(ss[:, 1] - base[:, 1])
        genuine = (d_dc[1] > 0) and (d_dc[0] - s_dc[0] > 0.01) and (d_ex[2] >= -0.005)
        verdict = "GENUINE" if genuine else ("re-alloc" if d_dc[1] > 0 else "ns")
        name = DISP.get(m, m)
        res[name] = {"orig": {"exact": float(base[:, 0].mean()), "disc": float(np.nanmean(base[:, 1])),
                              "conc": float(np.nanmean(base[:, 2]))},
                     "em_tuned_dEM": d_em, "disc_tuned": {"dexact": d_ex, "dconc": d_cc, "ddisc": d_dc,
                     "shuffle_ddisc": s_dc, "lam_med": float(np.median(lams))}, "verdict": verdict}
        print(f"{name:<14}{d_em[0]:>+9.3f}[{d_em[1]:+.2f}]   |{d_ex[0]:>+14.3f}{d_cc[0]:>+9.3f}"
              f"{d_dc[0]:>+9.3f}[{d_dc[1]:+.2f},{d_dc[2]:+.2f}]{np.median(lams):>5.1f}  {verdict}")
    json.dump(res, open(os.path.join(P1, "results/routing/E3_deconfound_simple.json"), "w"), indent=1)
    print("\nwrote results/routing/E3_deconfound_simple.json")


if __name__ == "__main__":
    main()
