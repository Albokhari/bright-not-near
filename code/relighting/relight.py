"""§16 shared engine — per-point counterfactual relighting operator.

A single smooth multiplicative gain field G(x,y) (Gaussian bumps at the 5 annotated points,
σ≈min(H,W)/6, partition-of-unity → 1 far from points) drives the whole program:
  - mode='adversarial'      brighten FAR points / darken NEAR  → DIAGNOSE shortcut (E1 probe)
  - mode='consistent'       brighten NEAR / darken FAR         → amplify the natural confound
  - mode='order_preserving' single GLOBAL gain (brightness changes, point ORDER does not) → the NULL
  - mode='random'           per-point random low-freq field    → TTA / consistency views (E3-adv, N4)
Relight runs BEFORE any model preprocessing, so no model code changes.
`point_brightness` copies finetune_dav2.load_item's exact 0.299R+0.587G+0.114B readout.
Self-test (python3 relight.py) certifies a RELATIVE brightness-order contrast (adversarial vs consistent swing
> 0.1 in ≥80% of images) — not full absolute inversion (at AMP=0.5 the adversarial relit image's brightness
order is fully inverted in ~25% of images; the downstream swing metric is purely relative, so this suffices).
order_preserving / uniform_bump leave the per-point order unchanged. NOTE: den-floor 0.10 + opposite-sign bump
overlap + display-space clipping make the realised per-point gain ≈0.78× nominal AMP → the reliance is a
CONSERVATIVE (lower-bound) estimate of brightness sensitivity.
"""
import numpy as np
import cv2

AMP = 0.5           # max multiplicative brightness change (±50%), applied in display space


def point_brightness(bgr, xy):
    """Grayscale luminance at points (xy: (N,2) original-res int coords). Matches load_item."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(float)
    g = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    xy = np.asarray(xy)
    bx = np.clip(xy[:, 0].astype(int), 0, g.shape[1] - 1)
    by = np.clip(xy[:, 1].astype(int), 0, g.shape[0] - 1)
    return g[by, bx]


def brightness_field(bgr, pts, mode, ranks=None, amp=AMP, seed=0):
    """Return a smooth multiplicative gain map G[H,W] (≈1 far from the 5 points)."""
    H, W = bgr.shape[:2]
    pts = np.asarray(pts, float)
    if mode == "order_preserving":                       # NULL: one global gain, no order change
        rng = np.random.default_rng(seed)
        return np.full((H, W), 1.0 + amp * (rng.random() * 2 - 1), np.float32)
    if mode == "random":
        rng = np.random.default_rng(seed)
        v = amp * (rng.random(len(pts)) * 2 - 1)
    elif mode == "uniform_bump":                         # NULL (S2): localized bumps, SAME signed gain at all
        v = np.full(len(pts), amp, float)                # 5 points → order-preserving but spatially structured
    elif mode == "scramble":                             # NULL (S3): adversarial gain magnitudes, RANDOM point
        r = np.asarray(ranks, float)                     # assignment → same footprint+magnitudes, breaks order link
        rn = 2 * (r - r.min()) / (r.max() - r.min() + 1e-9) - 1
        rng = np.random.default_rng(seed)
        v = rng.permutation(amp * rn)
    else:
        r = np.asarray(ranks, float)
        rn = 2 * (r - r.min()) / (r.max() - r.min() + 1e-9) - 1   # rank→[-1,1]; near(rank1)=-1, far=+1
        v = amp * rn if mode == "adversarial" else -amp * rn      # adversarial: darken near, brighten far
    sigma = min(H, W) / 6.0
    yy, xx = np.mgrid[0:H, 0:W]
    num = np.zeros((H, W), np.float32)
    den = np.full((H, W), 0.10, np.float32)              # background → G→1 far from points
    for (x, y), vi in zip(pts, v):
        b = np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))).astype(np.float32)
        num += vi * b
        den += b
    return (1.0 + num / den).astype(np.float32)


def apply(bgr, G):
    """Apply the multiplicative gain field in display space, clip to uint8."""
    return np.clip(bgr.astype(np.float32) * G[..., None], 0, 255).astype(np.uint8)


def relit(bgr, pts, mode, ranks=None, amp=AMP, seed=0):
    return apply(bgr, brightness_field(bgr, pts, mode, ranks, amp, seed))


def _selftest():
    import os, json
    from scipy.stats import kendalltau
    P1 = "/well/rittscher/users/fxh757/Code/Papers/P1_Ordinal_Endo_Depth_Benchmark__EndoLINA_MICCAI2026"
    IMG = "/well/rittscher/users/fxh757/Code/Kvasir_Ranking/Merged_PASS/images"
    g = json.load(open(os.path.join(P1, "results/consensus_GT/merged_pass_consensus_gt.json")))
    norig = nswing = nop = 0; n = 0; swings = []
    for im in g["images"][:60]:
        bgr = cv2.imread(os.path.join(IMG, im["filename"]))
        if bgr is None:
            continue
        pts = sorted(im["points"], key=lambda p: p["point_id"])
        xy = np.array([[p["x"], p["y"]] for p in pts], float)
        ranks = np.array([p["rank"] for p in pts], float)
        t0 = kendalltau(point_brightness(bgr, xy), ranks).correlation                       # natural (usually <0)
        tc = kendalltau(point_brightness(relit(bgr, xy, "consistent", ranks), xy), ranks).correlation   # more negative
        ta = kendalltau(point_brightness(relit(bgr, xy, "adversarial", ranks), xy), ranks).correlation  # less negative/positive
        to = kendalltau(point_brightness(relit(bgr, xy, "order_preserving", ranks, seed=1), xy), ranks).correlation
        norig += (t0 < 0); nswing += (ta > tc + 0.1); nop += (abs(to - t0) < 0.15)
        swings.append(ta - tc); n += 1
    print(f"selftest on {n} imgs: natural bright~near (t0<0) {norig}/{n} | "
          f"adversarial less-near than consistent {nswing}/{n} (mean swing {np.mean(swings):+.2f}) | "
          f"order_preserving keeps order {nop}/{n}")
    assert nswing >= 0.8 * n, "adversarial vs consistent relight does not create a brightness-order swing"
    assert nop >= 0.8 * n, "order_preserving relight changes the per-point brightness order"
    print("RELIGHT SELF-TEST PASSED")


if __name__ == "__main__":
    _selftest()
