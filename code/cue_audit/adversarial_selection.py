"""Benchmark v2 — brightness-adversarial point SELECTION on C3VD (annotator-free, dense GT).

Point selection is a controllable lever. We build, per frame, two 5-point sets from the dense GT:
  (a) RANDOM-quantile   : 5 depth-quantile points (the standard protocol).
  (b) ADVERSARIAL       : near points chosen DARK, far points chosen BRIGHT -> brightness order is maximally
                          anti-correlated with depth order (the shortcut is deliberately wrong).
The zero-parameter brightness baseline scores ~0.87 on (a) and ~0 on (b) by construction. A model that uses
real geometry stays high on (b); a brightness-reliant one drops. Adversarial selection is thus a harder,
higher-power split that separates models. We score models with dense C3VD predictions (DAV2, EndoOmni, DAC);
orientation is auto-calibrated per model from the random set.
"""
import os, glob, numpy as np, cv2
from PIL import Image

C3 = "/well/rittscher/users/fxh757/Datasets/PUBLIC_ENDO/Depth_split_2/test"
P1 = "/well/rittscher/users/fxh757/Code/Papers/P1_Ordinal_Endo_Depth_Benchmark__EndoLINA_MICCAI2026"
OUT = os.path.join(P1, "results/validation/adversarial"); os.makedirs(OUT, exist_ok=True)
MODELS = {"DAV2-base": "results/finetune/c3vd_preds/base/pred_depth_u16",
          "DAV2-ordinalFT": "results/finetune/c3vd_preds/dav2_fold0_ft/pred_depth_u16"}
rng = np.random.default_rng(0)
FOV = None


def fov(hw):
    global FOV
    if FOV is None or FOV.shape != hw:
        m = np.asarray(Image.open(os.path.join(C3, "invalid_mask.png")).convert("L"))
        FOV = np.asarray(Image.fromarray(m).resize((hw[1], hw[0]), Image.NEAREST)) > 127
    return FOV


def pick(dep, bright, mode, k=5, ncand=400, min_frac=0.06):
    ys, xs = np.where((dep > 0) & fov(dep.shape))
    if len(ys) < 200:
        return None
    H, W = dep.shape; mind = min_frac * np.hypot(H, W)
    d = dep[ys, xs].astype(float); b = bright[ys, xs].astype(float)
    qd = np.quantile(d, np.linspace(0.10, 0.90, k))           # near -> far target depths
    # for adversarial: near quantile wants LOW brightness, far wants HIGH brightness
    qb_targets = np.quantile(b, np.linspace(0.10, 0.90, k)) if mode == "adv" else [None] * k
    chosen = []
    for kk in range(k):
        cand = rng.choice(len(ys), size=min(ncand, len(ys)), replace=False)
        if mode == "adv":
            # candidates near the depth quantile, ranked by how well brightness matches the ADVERSARIAL target
            near_d = cand[np.abs(d[cand] - qd[kk]) < (d.max() - d.min()) * 0.12 + 1e-6]
            if len(near_d) < 3:
                near_d = cand[np.argsort(np.abs(d[cand] - qd[kk]))[:20]]
            cand = near_d[np.argsort(np.abs(b[near_d] - qb_targets[kk]))]
        else:
            cand = cand[np.argsort(np.abs(d[cand] - qd[kk]))]
        placed = False
        for ci in cand:
            p = np.array([xs[ci], ys[ci]], float)
            if all(np.hypot(*(p - c)) > mind for c in chosen):
                chosen.append(p); placed = True; break
        if not placed:
            chosen.append(np.array([xs[cand[0]], ys[cand[0]]], float))
    xy = np.array(chosen).astype(int)
    dv = dep[xy[:, 1], xy[:, 0]].astype(float)
    if len(np.unique(dv)) < k:
        return None
    return xy, dv


def pairwise(scores, dv, near_is_large):
    s = scores if near_is_large else -scores
    ok = tot = 0
    for i in range(5):
        for j in range(5):
            if dv[i] < dv[j]:                       # i nearer (smaller depth)
                tot += 1; ok += int(s[i] > s[j])
    return ok, tot


def main():
    imgs = sorted(glob.glob(os.path.join(C3, "images", "*.png")))[::6]     # ~480 frames
    preds = {m: {os.path.basename(p): p for p in glob.glob(os.path.join(P1, d, "*.png"))}
             for m, d in MODELS.items()}
    # accumulate per-frame (fn -> {set -> (xy, dv, bright_at_pts)})
    acc = {m: {"rand": [0, 0], "adv": [0, 0]} for m in list(MODELS) + ["Brightness"]}
    signs = {m: None for m in MODELS}
    nf = 0
    for ip in imgs:
        fn = os.path.basename(ip)
        dep = np.asarray(Image.open(os.path.join(C3, "depths", fn))).astype(np.float64)
        bgr = cv2.imread(ip)
        if bgr is None or dep.ndim != 2:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(float)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        gray = cv2.resize(gray, (dep.shape[1], dep.shape[0]))
        sets = {}
        for mode in ("rand", "adv"):
            r = pick(dep, gray, "adv" if mode == "adv" else "rand")
            if r is None:
                sets = None; break
            sets[mode] = r
        if sets is None:
            continue
        nf += 1
        for mode in ("rand", "adv"):
            xy, dv = sets[mode]
            H, W = dep.shape
            bpt = gray[xy[:, 1], xy[:, 0]]
            ok, tot = pairwise(bpt, dv, near_is_large=True)     # brightness baseline (bright=near) -- no preds needed
            acc["Brightness"][mode][0] += ok; acc["Brightness"][mode][1] += tot
            for m in MODELS:
                if fn not in preds[m]:                          # per-model skip (don't abort the frame)
                    continue
                pd = np.asarray(Image.open(preds[m][fn])).astype(float)
                if pd.ndim == 3:
                    pd = pd[..., 0]
                pd = cv2.resize(pd, (W, H), interpolation=cv2.INTER_NEAREST)
                sc = pd[xy[:, 1], xy[:, 0]]
                if signs[m] is None:                            # calibrate on the random set: pick sign giving higher pairwise
                    o1, t1 = pairwise(sc, dv, True); o2, t2 = pairwise(sc, dv, False)
                    signs[m] = (o1 / max(t1, 1)) >= (o2 / max(t2, 1))
                ok, tot = pairwise(sc, dv, signs[m])
                acc[m][mode][0] += ok; acc[m][mode][1] += tot
    print(f"scored {nf} C3VD frames | models {list(MODELS)}\n")
    print(f"{'model':<12}{'random':>10}{'adversarial':>13}{'drop':>8}")
    lines = []
    for m in ["Brightness"] + list(MODELS):
        pr = acc[m]["rand"][0] / max(acc[m]["rand"][1], 1)
        pa = acc[m]["adv"][0] / max(acc[m]["adv"][1], 1)
        print(f"{m:<12}{pr:>10.3f}{pa:>13.3f}{pr-pa:>+8.3f}")
        lines.append(f"{m},{pr:.4f},{pa:.4f},{pr-pa:.4f}")
    open(os.path.join(OUT, "adversarial_selection.csv"), "w").write(
        "model,random,adversarial,drop\n" + "\n".join(lines) + "\n")
    print(f"\nwrote {OUT}/adversarial_selection.csv")


if __name__ == "__main__":
    main()
