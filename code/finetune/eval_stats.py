"""Decision-gate statistics for the fine-tuning pilot (and full CV).

Reads one or more <run>/test_points.csv (per filename/point: gt_rank, brightness,
base_score, ft_score) and computes, per image, base & finetuned ordinal metrics,
then the IMAGE-CLUSTERED bootstrap 95% CI on the finetuned-minus-base delta for
ExactMatch and brightness-adversarial (discordant) accuracy. Gate = discordant
delta CI clear of 0.

Usage: python3 eval_stats.py <run1> [<run2> ...]   (multiple = pooled CV folds)
"""
import os, sys, csv
import numpy as np

P1 = "/well/rittscher/users/fxh757/Code/Papers/P1_Ordinal_Endo_Depth_Benchmark__EndoLINA_MICCAI2026"


def pairs(r):
    return [(i, j) for i in range(5) for j in range(5) if r[i] < r[j]]


def per_image(scores, ranks, bright):
    s, r, b = np.asarray(scores), np.asarray(ranks), np.asarray(bright)
    exact = float(np.array_equal(np.argsort(-s), np.argsort(r)))
    dc = dn = 0
    for (i, j) in pairs(r):
        if b[i] < b[j]:                 # brightness-adversarial pair
            dc += int(s[i] > s[j]); dn += 1
    return exact, dc, dn


def load(runs):
    imgs = {}
    for run in runs:
        path = os.path.join(P1, "results/finetune", run, "test_points.csv")
        cur = {}
        for row in csv.DictReader(open(path)):
            fn = row["filename"]
            cur.setdefault(fn, {"r": [0]*5, "b": [0]*5, "base": [0]*5, "ft": [0]*5})
            k = int(row["point_idx"])
            cur[fn]["r"][k] = float(row["gt_rank"]); cur[fn]["b"][k] = float(row["brightness"])
            cur[fn]["base"][k] = float(row["base_score"]); cur[fn]["ft"][k] = float(row["ft_score"])
        imgs.update(cur)   # folds are image-disjoint
    return imgs


def main():
    runs = sys.argv[1:] or ["pilot_dav2"]
    imgs = load(runs)
    rng = np.random.default_rng(0)
    be, bdc, bdn, fe, fdc, fdn = [], [], [], [], [], []
    for fn, d in imgs.items():
        e0, c0, n0 = per_image(d["base"], d["r"], d["b"])
        e1, c1, n1 = per_image(d["ft"], d["r"], d["b"])
        be.append(e0); fe.append(e1); bdc.append(c0); fdc.append(c1); bdn.append(n0); fdn.append(n1)
    be, fe = np.array(be), np.array(fe)
    bdc, fdc, dn = np.array(bdc, float), np.array(fdc, float), np.array(bdn, float)
    n = len(be)
    base_exact, ft_exact = be.mean(), fe.mean()
    base_disc = bdc.sum() / dn.sum(); ft_disc = fdc.sum() / dn.sum()

    B = 10000; idx = rng.integers(0, n, size=(B, n))
    d_exact = (fe - be)[idx].mean(1)
    num = (fdc - bdc)[idx].sum(1); den = dn[idx].sum(1)
    d_disc = num / den
    def ci(x): return np.percentile(x, [2.5, 97.5])
    ex_lo, ex_hi = ci(d_exact); di_lo, di_hi = ci(d_disc)

    print(f"images={n}  runs={runs}")
    print(f"ExactMatch:  base {base_exact:.3f} -> ft {ft_exact:.3f}  Δ {ft_exact-base_exact:+.3f} [{ex_lo:+.3f},{ex_hi:+.3f}]")
    print(f"Discordant:  base {base_disc:.3f} -> ft {ft_disc:.3f}  Δ {ft_disc-base_disc:+.3f} [{di_lo:+.3f},{di_hi:+.3f}]")
    gate = di_lo > 0
    print(f"GATE (discordant Δ CI clear of 0): {'PASS' if gate else 'not passed'}")


if __name__ == "__main__":
    main()
