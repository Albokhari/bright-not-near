"""Exp A metrics: per-image scale+shift alignment + dense metric-depth errors.

Reuses the alignment + metric conventions of
  code/c3vd_eval/eval_c3vd_relative.py
verbatim where possible:
  - ss alignment = least-squares scale+shift  gt ~= a*pred + b  (A = [pred, 1]),
    a is allowed to be NEGATIVE so a disparity-like prediction can align to depth.
  - masked pixels only; prediction clipped to [1e-3, +inf) before ratio/log metrics.
  - GT is metric depth in mm (larger = farther); valid mask is 0 < gt <= 100mm
    (built by the caller; this module just consumes the boolean mask).

Contract interface (exact names):
  ss_fit(pred HxW, gt HxW, mask HxW) -> (a, b)        # least-squares scale+shift
  apply(pred, a, b) -> aligned                        # a*pred + b
  metrics(aligned HxW, gt HxW, mask HxW) -> dict{absrel, rmse, silog, grad, d125}

Beyond eval_c3vd_relative this ADDS:
  - silog : scale-invariant log RMSE (Eigen), sqrt(mean(d^2) - mean(d)^2), d=log e - log G.
  - grad  : mean L1 depth-gradient error via Sobel, |dPx-dGx| + |dPy-dGy| averaged over
            the interior of the valid mask (3x3-eroded so every Sobel window is valid).

Pure numpy (scipy not required). Sobel + mask erosion are implemented with numpy
3x3 correlations so the module imports even in a scipy-free environment.

Self-test: python3 expA_metrics.py
"""
import numpy as np

EPS = 1e-6
PRED_FLOOR = 1e-3          # matches eval_c3vd_relative.metrics clip

# Sobel kernels (correlation form).  dx = horizontal gradient (axis=1),
# dy = vertical gradient (axis=0).
_KX = np.array([[-1., 0., 1.],
                [-2., 0., 2.],
                [-1., 0., 1.]], dtype=np.float64)
_KY = np.array([[-1., -2., -1.],
                [0.,  0.,  0.],
                [1.,  2.,  1.]], dtype=np.float64)


# --------------------------------------------------------------------------- #
# alignment
# --------------------------------------------------------------------------- #
def ss_fit(pred, gt, mask):
    """Least-squares scale+shift so that gt ~= a*pred + b on masked pixels.

    Returns (a, b) as python floats.  `a` may be negative (disparity vs depth).
    Mirrors eval_c3vd_relative.align(..., mode="ss") but returns the coeffs so
    the caller can FREEZE them on the original frame and reuse on relit frames.
    """
    P = np.asarray(pred, dtype=np.float64)
    G = np.asarray(gt, dtype=np.float64)
    m = np.asarray(mask, dtype=bool) & np.isfinite(P) & np.isfinite(G)   # exclude NaN pred (crop-model FOV margins)
    P = P[m]; G = G[m]
    if P.size < 2:
        return 1.0, 0.0
    A = np.stack([P, np.ones_like(P)], axis=1)
    sol, *_ = np.linalg.lstsq(A, G, rcond=None)
    a, b = sol
    if not (np.isfinite(a) and np.isfinite(b)):
        return 1.0, 0.0
    return float(a), float(b)


def apply(pred, a, b):
    """Apply the frozen affine transform: a*pred + b."""
    return a * np.asarray(pred, dtype=np.float64) + b


# --------------------------------------------------------------------------- #
# Sobel gradient error (pure-numpy)
# --------------------------------------------------------------------------- #
def _corr3(img, k):
    """3x3 correlation with edge padding (pure numpy)."""
    p = np.pad(img, 1, mode="edge")
    out = np.zeros_like(img, dtype=np.float64)
    H, W = img.shape
    for di in range(3):
        for dj in range(3):
            w = k[di, dj]
            if w != 0.0:
                out += w * p[di:di + H, dj:dj + W]
    return out


def _erode3(mask):
    """3x3 binary erosion: True only where the full 3x3 neighborhood is inside
    the mask.  Guarantees every Sobel 3x3 window sits on valid pixels, so the
    gt=0 border (outside FOV / invalid depth) never contaminates the gradient."""
    m = np.asarray(mask, dtype=bool)
    p = np.pad(m, 1, mode="constant", constant_values=False)
    out = np.ones_like(m, dtype=bool)
    H, W = m.shape
    for di in range(3):
        for dj in range(3):
            out &= p[di:di + H, dj:dj + W]
    return out


def grad_error(aligned, gt, mask):
    """Mean L1 depth-gradient error via Sobel, averaged over the mask interior."""
    a = np.asarray(aligned, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    ax, ay = _corr3(a, _KX), _corr3(a, _KY)
    gx, gy = _corr3(g, _KX), _corr3(g, _KY)
    err = np.abs(ax - gx) + np.abs(ay - gy)
    m = _erode3(mask)
    if m.sum() == 0:                       # tiny/thin mask -> fall back
        m = np.asarray(mask, dtype=bool)
    if m.sum() == 0:
        return 0.0
    return float(err[m].mean())


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def metrics(aligned, gt, mask):
    """Dense depth errors of an already-ss-aligned prediction vs GT (mm).

    Returns dict{absrel, rmse, silog, grad, d125}.
      absrel : mean |e-G|/G
      rmse   : sqrt(mean (e-G)^2)   (mm, since GT is mm)
      silog  : scale-invariant log RMSE = sqrt(mean d^2 - mean(d)^2), d=log e-log G
      grad   : mean L1 Sobel depth-gradient error over the mask interior
      d125   : fraction of pixels with max(e/G, G/e) < 1.25
    """
    al = np.asarray(aligned, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    m = np.asarray(mask, dtype=bool) & np.isfinite(al) & np.isfinite(g)   # exclude NaN pred (crop-model FOV margins)

    e = np.clip(al[m], PRED_FLOOR, None)
    G = g[m]
    valid = G > EPS
    e, G = e[valid], G[valid]
    if e.size == 0:
        return dict(absrel=float("nan"), rmse=float("nan"),
                    silog=float("nan"), grad=float("nan"), d125=float("nan"))

    ar = np.abs(e - G) / G
    rat = np.maximum(e / G, G / e)
    d = np.log(e) - np.log(G)
    silog = float(np.sqrt(max(np.mean(d * d) - np.mean(d) ** 2, 0.0)))

    return dict(
        absrel=float(ar.mean()),
        rmse=float(np.sqrt(np.mean((e - G) ** 2))),
        silog=silog,
        grad=grad_error(al, g, m),
        d125=float(np.mean(rat < 1.25)),
    )


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def _make_scene(H=48, W=64, seed=0):
    """A smooth positive depth field (mm) with real spatial gradients + a mostly
    valid mask (a few invalid/border pixels)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    ramp = 20.0 + 60.0 * (xx / (W - 1))                      # 20..80 mm ramp
    bump = 15.0 * np.exp(-(((xx - W * 0.6) ** 2 + (yy - H * 0.4) ** 2) / (2 * 80.0)))
    gt = ramp + bump                                         # ~20..95 mm, smooth
    mask = np.ones((H, W), dtype=bool)
    mask[:2, :] = mask[-2:, :] = mask[:, :2] = mask[:, -2:] = False   # border invalid
    mask[gt <= 0] = False
    return gt, mask, rng


def _selftest():
    ok = True
    gt, mask, rng = _make_scene()

    def show(tag, pred):
        a, b = ss_fit(pred, gt, mask)
        al = apply(pred, a, b)
        r = metrics(al, gt, mask)
        print(f"  {tag:14s} a={a:+8.4f} b={b:+9.4f} | "
              f"absrel={r['absrel']:.4e} rmse={r['rmse']:.4e} "
              f"silog={r['silog']:.4e} grad={r['grad']:.4e} d125={r['d125']:.4f}")
        assert all(np.isfinite(v) for v in r.values()), f"{tag}: non-finite metric"
        return a, b, r

    print("expA_metrics self-test")

    # 1) perfect prediction -> ~0 error, d125==1, a~1 b~0
    _, _, r = show("perfect", gt.copy())
    ok &= r["absrel"] < 1e-6 and r["rmse"] < 1e-6 and r["silog"] < 1e-6
    ok &= r["grad"] < 1e-6 and abs(r["d125"] - 1.0) < 1e-9

    # 2) affine-distorted prediction (positive scale) -> ss recovers, ~0 error
    a2, b2, r = show("affine 3x+5", 3.0 * gt + 5.0)
    ok &= abs(a2 - (1.0 / 3.0)) < 1e-6 and r["absrel"] < 1e-6 and r["rmse"] < 1e-6

    # 3) inverted / disparity-like prediction -> NEGATIVE a recovered, ~0 error
    a3, b3, r = show("inverted", 100.0 - gt)
    ok &= a3 < 0 and r["absrel"] < 1e-6 and r["rmse"] < 1e-6

    # 4) noisy prediction -> finite, non-trivial error; d125 in (0,1]
    _, _, r = show("noisy", gt + rng.normal(0, 4.0, gt.shape))
    ok &= r["absrel"] > 0 and r["rmse"] > 0 and 0.0 < r["d125"] <= 1.0

    # 5) grad responds to a gradient-only perturbation (add a checkerboard that
    #    barely moves absrel but spikes the Sobel error)
    board = np.indices(gt.shape).sum(0) % 2
    _, _, r_clean = show("smooth base", gt.copy())
    _, _, r_edgy = show("edgy", gt + 2.0 * board)
    ok &= r_edgy["grad"] > r_clean["grad"]

    # 6) empty mask -> NaNs, no crash
    r = metrics(gt, gt, np.zeros_like(mask))
    ok &= all(np.isnan(v) for v in r.values())
    print(f"  empty-mask     -> {r}")

    print("SELF-TEST:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
