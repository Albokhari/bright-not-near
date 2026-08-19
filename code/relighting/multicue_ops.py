"""PILLAR 5 (multi-cue) — counterfactual operators BEYOND brightness: specular glint + defocus blur, plus a
global vignette used as a control. Each targets a NON-brightness nuisance cue at the 5 annotated points in an
order-ADVERSARIAL vs order-CONSISTENT way, mirroring the brightness relight (relight.py), so the same swing /
null machinery applies. Geometry is preserved: we add wet-surface glints and local defocus (real endoscopic
nuisances) — we do NOT move points or repaint shape. A human-verification subset (e_multicue.py) audits that.

Cue proxies (proxy()) let the self-test certify that adversarial vs consistent actually swings the cue order,
exactly as relight._selftest does for brightness.
  specular: fraction of near-white pixels in a patch (wet glint)   -> adversarial adds glints on FAR points
  blur    : variance-of-Laplacian sharpness in a patch             -> adversarial blurs NEAR points
"""
import numpy as np
import cv2

AMP = 0.5


def _rank_norm(ranks):
    r = np.asarray(ranks, float)
    return 2 * (r - r.min()) / (r.max() - r.min() + 1e-9) - 1    # near(rank1) -> -1, far -> +1


def _blobs(H, W, xy, sigma):
    yy, xx = np.mgrid[0:H, 0:W]
    return [np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))).astype(np.float32) for (x, y) in xy]


def _weights(ranks, mode, amp, seed, target_far):
    """Per-point magnitudes. target_far=True => the 'adversarial' arm loads the FAR points (specular);
    False => adversarial loads the NEAR points (blur). consistent is the opposite arm; matched permutes the
    adversarial magnitudes (same footprint, broken order link); away uses adversarial magnitudes off-points."""
    rn = _rank_norm(ranks)
    w_far = (rn + 1) / 2.0            # 0 at nearest, 1 at farthest
    w_near = 1.0 - w_far
    adv = w_far if target_far else w_near
    con = w_near if target_far else w_far
    if mode == "adversarial":
        v = amp * adv
    elif mode == "consistent":
        v = amp * con
    elif mode == "matched":
        rng = np.random.default_rng(seed)
        v = rng.permutation(amp * adv)
    elif mode == "away":
        v = amp * adv
    else:
        raise ValueError(mode)
    return v


def specular_apply(bgr, xy, ranks, mode, amp=AMP, seed=0):
    """Add saturated white glints as small Gaussian disks at the points (alpha-blend toward pure white so the
    core clears the near-white threshold regardless of the base intensity)."""
    H, W = bgr.shape[:2]
    sigma = max(2.0, min(H, W) / 45.0)
    pts = _away_pts(H, W, len(ranks), seed) if mode == "away" else np.asarray(xy, float)
    v = _weights(ranks, mode, amp, seed, target_far=True)
    a = np.zeros((H, W), np.float32)
    for b, vi in zip(_blobs(H, W, pts, sigma), v):
        a = np.maximum(a, np.clip(vi * 2.0, 0, 1) * b)     # vi>=0.5 -> pure-white core (glint)
    a = np.clip(a, 0, 1)[..., None]
    return (bgr.astype(np.float32) * (1 - a) + 255.0 * a).clip(0, 255).astype(np.uint8)


def blur_apply(bgr, xy, ranks, mode, amp=AMP, seed=0):
    """Local defocus: composite a globally blurred copy in soft patches at the points."""
    H, W = bgr.shape[:2]
    blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=max(1.0, min(H, W) / 40.0))
    sigma = max(3.0, min(H, W) / 12.0)
    pts = _away_pts(H, W, len(ranks), seed) if mode == "away" else np.asarray(xy, float)
    v = _weights(ranks, mode, amp, seed, target_far=False)
    m = np.zeros((H, W), np.float32)
    for b, vi in zip(_blobs(H, W, pts, sigma), v):
        m = np.maximum(m, vi * b)                          # union of per-point defocus masks
    m = np.clip(m, 0, 1)[..., None]
    return (bgr.astype(np.float32) * (1 - m) + blurred.astype(np.float32) * m).clip(0, 255).astype(np.uint8)


def vignette_apply(bgr, strength=0.4, invert=False):
    """CONTROL: global radial illumination gradient (peripheral darkening, or center darkening if invert).
    Smooth + point-order-agnostic, so a geometry-using model must NOT swing on it."""
    H, W = bgr.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    rho = np.sqrt(((xx - cx) / (W / 2.0)) ** 2 + ((yy - cy) / (H / 2.0)) ** 2)
    rho = np.clip(rho / rho.max(), 0, 1)
    g = (1 - strength * rho ** 2) if not invert else (1 - strength * (1 - rho) ** 2)
    return np.clip(bgr.astype(np.float32) * g[..., None], 0, 255).astype(np.uint8)


def _away_pts(H, W, n, seed):
    """n random off-center locations (control: apply the perturbation away from the annotated points)."""
    rng = np.random.default_rng(1000 + seed)
    xs = rng.integers(int(0.15 * W), int(0.85 * W), n)
    ys = rng.integers(int(0.15 * H), int(0.85 * H), n)
    return np.stack([xs, ys], 1).astype(float)


OPS = {"specular": specular_apply, "blur": blur_apply}


def apply(cue, bgr, xy, ranks, mode, amp=AMP, seed=0):
    return OPS[cue](bgr, xy, ranks, mode, amp=amp, seed=seed)


# ---- cue proxies (for the self-test only; the downstream metric is the model's pairwise swing) ----
def _patch(gray, x, y, h):
    H, W = gray.shape
    x, y = int(x), int(y)
    return gray[max(0, y - h):min(H, y + h + 1), max(0, x - h):min(W, x + h + 1)]


def proxy(cue, bgr, xy, h=9):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    vals = []
    for (x, y) in np.asarray(xy, float):
        p = _patch(g, x, y, h)
        if cue == "specular":
            vals.append(float((p > 0.95 * 255).mean()))                    # wet-glint fraction
        elif cue == "blur":
            vals.append(float(cv2.Laplacian(p, cv2.CV_32F).var()))         # sharpness (higher = sharper)
    return np.array(vals)


def _selftest():
    import os, json
    from scipy.stats import kendalltau
    P1 = "/well/rittscher/users/fxh757/Code/Papers/P1_Ordinal_Endo_Depth_Benchmark__EndoLINA_MICCAI2026"
    IMG = "/well/rittscher/users/fxh757/Code/Kvasir_Ranking/Merged_PASS/images"
    g = json.load(open(os.path.join(P1, "results/consensus_GT/merged_pass_consensus_gt.json")))
    stats = {c: {"swing": [], "adv_gt_con": 0, "valid": 0} for c in OPS}
    for im in g["images"][:80]:
        bgr = cv2.imread(os.path.join(IMG, im["filename"]))
        if bgr is None:
            continue
        pts = sorted(im["points"], key=lambda p: p["point_id"])
        xy = np.array([[p["x"], p["y"]] for p in pts], float)
        ranks = np.array([p["rank"] for p in pts], float)
        for c in OPS:
            adv = proxy(c, apply(c, bgr, xy, ranks, "adversarial"), xy)
            con = proxy(c, apply(c, bgr, xy, ranks, "consistent"), xy)
            ta = kendalltau(adv, ranks).correlation
            tc = kendalltau(con, ranks).correlation
            if not (np.isnan(ta) or np.isnan(tc)):
                stats[c]["swing"].append(ta - tc)
                stats[c]["adv_gt_con"] += int(ta > tc + 0.05)
                stats[c]["valid"] += 1
    for c in OPS:
        sw = np.nanmean(stats[c]["swing"]); nv = stats[c]["valid"]; frac = stats[c]["adv_gt_con"] / max(nv, 1)
        print(f"{c:9s}: mean cue-order swing (adv−con) {sw:+.3f} | adv>con in {stats[c]['adv_gt_con']}/{nv} ({frac:.0%})")
        assert frac >= 0.7, f"{c}: adversarial vs consistent does not swing the cue order in >=70% of valid images"
    print("MULTICUE-OPS SELF-TEST PASSED")


if __name__ == "__main__":
    _selftest()
