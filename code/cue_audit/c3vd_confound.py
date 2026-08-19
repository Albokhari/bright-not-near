"""C3VD cross-dataset confound reproduction (annotator-free).

Shows the near-field "bright = near" shortcut is NOT a Kvasir artifact: on C3VD
(a different dataset, with dense metric GT) a zero-parameter inverse-brightness
baseline also achieves high ordinal-correlation scores against the true geometric
order, and a sizable fraction of point-pairs are brightness-adversarial.

Uses only C3VD RGB + GT depth (no model predictions, no annotators):
  - GT order at 5 sampled points (near = smaller depth value)
  - brightness order = grayscale intensity (near = brighter)
Reports brightness-baseline ordinal metrics vs GT + discordant fraction +
Spearman(brightness, geometry), with image-clustered bootstrap CIs.

Output: results/validation/v2/c3vd_confound.{csv,md}
"""
import os
import glob
import numpy as np
from PIL import Image
from scipy.stats import kendalltau, spearmanr
import p1_common as C

C3 = "/well/rittscher/users/fxh757/Datasets/PUBLIC_ENDO/Depth_split_2/test"
OUT = os.path.join(C.P1, "results/validation/v2")
N_IMG = 300
K_SETS = 40
SEED = 0


def gray(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def sign(x):
    return int(x > 0) - int(x < 0)


def main():
    rng = np.random.default_rng(SEED)
    imgs = sorted(glob.glob(os.path.join(C3, "images", "*")))
    deps = sorted(glob.glob(os.path.join(C3, "depths", "*")))
    assert len(imgs) == len(deps)
    pick = rng.choice(len(imgs), size=min(N_IMG, len(imgs)), replace=False)

    # per-image tallies for the brightness baseline vs GT order
    per_img_all_c, per_img_all_n = [], []
    per_img_dis_n = []
    per_img_pair_c, per_img_pair_n = [], []   # pairwise accuracy of brightness baseline
    per_img_tau = []                          # tau(brightness, geometry) pooled per image
    exact_hits, exact_tot = [], []

    for idx in pick:
        rgb = np.asarray(Image.open(imgs[idx]).convert("RGB"), float)
        dep = np.asarray(Image.open(deps[idx])).astype(float)
        if rgb.shape[:2] != dep.shape[:2]:
            rgb = np.asarray(Image.open(imgs[idx]).convert("RGB").resize(
                (dep.shape[1], dep.shape[0])), float)
        g = gray(rgb)
        ys, xs = np.where(dep > 0)
        if len(ys) < 50:
            continue
        allc = alln = pc = pn = disn = 0
        eh = et = 0
        tb, tg = [], []
        for _ in range(K_SETS):
            sel = rng.choice(len(ys), size=5, replace=False)
            py, px = ys[sel], xs[sel]
            d = dep[py, px]          # GT depth value (near = small)
            b = g[py, px]            # brightness   (near = bright = large)
            if len(np.unique(d)) < 5:
                continue
            tb.extend(b.tolist()); tg.extend(d.tolist())
            for i in range(5):
                for j in range(i + 1, 5):
                    if d[i] == d[j]:
                        continue
                    sg = sign(d[i] - d[j])            # +1 if i farther than j
                    sb = sign(b[j] - b[i])            # brightness predicts: brighter=nearer
                    # brightness baseline correct iff its near/far ordering matches GT
                    alln += 1; pn += 1
                    correct = (sb == sg)
                    allc += correct; pc += correct
                    if sb != 0 and sb != sg:
                        disn += 1
            # exact-match of brightness ordering vs GT ordering (full 5)
            gt_order = np.argsort(d)                  # near->far indices
            br_order = np.argsort(-b)                 # bright->dark = near->far
            et += 1; eh += int(np.array_equal(gt_order, br_order))
        if alln == 0:
            continue
        per_img_all_c.append(allc); per_img_all_n.append(alln)
        per_img_pair_c.append(pc); per_img_pair_n.append(pn)
        per_img_dis_n.append(disn)
        exact_hits.append(eh); exact_tot.append(et)
        if len(set(tg)) > 2:
            per_img_tau.append(kendalltau(tb, tg).correlation)

    allc = np.array(per_img_all_c, float); alln = np.array(per_img_all_n, float)
    disn = np.array(per_img_dis_n, float)
    eh = np.array(exact_hits, float); et = np.array(exact_tot, float)
    taus = np.array([t for t in per_img_tau if np.isfinite(t)], float)

    def ratio_ci(c, n, seed=3, B=10000):
        rng2 = np.random.default_rng(seed)
        k = len(c); ii = rng2.integers(0, k, size=(B, k))
        boot = c[ii].sum(1) / n[ii].sum(1)
        return c.sum() / n.sum(), *np.percentile(boot, [2.5, 97.5])

    pair_acc, plo, phi = ratio_ci(allc, alln)
    disc_frac = disn.sum() / alln.sum()
    exact, elo, ehi = ratio_ci(eh, et, seed=4)
    tau_mean, tlo, thi = C.cluster_bootstrap_ci(taus, seed=5)

    lines = []
    lines.append("# C3VD cross-dataset confound reproduction (annotator-free, n=%d images x %d point-sets)\n" % (len(allc), K_SETS))
    lines.append("Zero-parameter inverse-brightness baseline scored against C3VD metric-GT depth order.\n")
    lines.append("| quantity | value [95%% CI] |")
    lines.append("|---|---|")
    lines.append(f"| Brightness-baseline pairwise accuracy vs GT | **{pair_acc:.3f}** [{plo:.3f}, {phi:.3f}] |")
    lines.append(f"| Brightness-baseline 5-pt ExactMatch vs GT | {exact:.3f} [{elo:.3f}, {ehi:.3f}] |")
    lines.append(f"| Kendall tau(brightness, geometry) | **{tau_mean:+.3f}** [{tlo:+.3f}, {thi:+.3f}] |")
    lines.append(f"| Brightness-adversarial (discordant) pair fraction | {disc_frac:.3f} |")
    lines.append("")
    lines.append("**Finding:** the bright=near shortcut reproduces on C3VD — a trivial brightness")
    lines.append("baseline attains pairwise accuracy %.2f and Kendall tau %+.2f against true metric" % (pair_acc, tau_mean))
    lines.append("geometry, with ~%.0f%% brightness-adversarial pairs. The confound is a property of" % (100*disc_frac))
    lines.append("near-field endoluminal illumination, not of the Kvasir benchmark.")
    md = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "c3vd_confound.md"), "w") as f:
        f.write(md)
    import csv
    with open(os.path.join(OUT, "c3vd_confound.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "ci_lo", "ci_hi"])
        w.writerow(["brightness_pairwise_vs_gt", pair_acc, plo, phi])
        w.writerow(["brightness_exact_vs_gt", exact, elo, ehi])
        w.writerow(["kendall_tau_brightness_geometry", tau_mean, tlo, thi])
        w.writerow(["discordant_fraction", disc_frac, "", ""])
    print(md)


if __name__ == "__main__":
    main()
