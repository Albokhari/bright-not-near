"""Exp A: dense, GT-depth-keyed, geometry-preserving relight families (unit = image/scene).
Each returns a relit BGR uint8 image the SAME size as the input; geometry (the GT depth map) is unchanged.
Families:
  global    F1  global radiometric (exposure / gamma / white-balance) -- depth-independent
  spatial   F2  smooth low-frequency spatial gain field (radial / vignette / Fourier), mean~1 -- depth-independent
  physics   F3  near-field physically-motivated relight (depth->normals->inverse-square Lambertian), light moved
  decorr    F4  brightness DECORRELATED from depth (depth-conditional luminance equalization -> |rho(L,D)|<0.05)
  invert    F5  brightness INVERTED vs depth (cue conflict: brighten far / darken near -> rho(L,nearness)<-0.3)
All gains applied in display (gamma-encoded) space then clipped to uint8, matching relight.apply.
`param` selects the realization/strength; `seed` varies stochastic fields.
"""
import os, sys
import numpy as np
import cv2

HERE = os.path.dirname(__file__)
_DM = "/well/rittscher/users/fxh757/Code/Depth_metrics_model"
if _DM not in sys.path:
    sys.path.insert(0, _DM)

FAMILIES = ["global", "spatial", "physics", "decorr", "invert"]


def _lum(bgr):
    b, g, r = bgr[..., 0].astype(np.float32), bgr[..., 1].astype(np.float32), bgr[..., 2].astype(np.float32)
    return 0.114 * b + 0.587 * g + 0.299 * r


def _apply_gain(bgr, gain):
    """Display-space multiplicative gain (HxW), clipped to uint8 (matches relight.apply)."""
    return np.clip(bgr.astype(np.float32) * gain[..., None], 0, 255).astype(np.uint8)


def _tissue_mask(bgr, depth):
    """Valid tissue pixels: positive GT depth AND not pure FOV-border black."""
    m = (depth > 0) & (_lum(bgr) > 3)
    return m


# ---------- F1 global radiometric ----------
def _f1_global(bgr, depth, param, seed):
    rng = np.random.default_rng(seed)
    kind = ["exposure", "gamma", "wb"][int(param) % 3] if param >= 1 else "exposure"
    x = bgr.astype(np.float32)
    if kind == "exposure":
        s = rng.uniform(0.5, 2.0)
        return np.clip(x * s, 0, 255).astype(np.uint8)
    if kind == "gamma":
        gm = rng.uniform(0.7, 1.5)
        return np.clip(255.0 * (x / 255.0) ** gm, 0, 255).astype(np.uint8)
    # white balance: per-channel BGR multipliers
    mult = rng.uniform(0.8, 1.2, size=3).astype(np.float32)
    return np.clip(x * mult[None, None, :], 0, 255).astype(np.uint8)


# ---------- F2 smooth spatial field (mean ~ 1, depth-independent) ----------
def _f2_spatial(bgr, depth, param, seed, amp=0.4):
    H, W = bgr.shape[:2]
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cy, cx = H / 2.0, W / 2.0
    rho = np.sqrt(((xx - cx) / (W / 2)) ** 2 + ((yy - cy) / (H / 2)) ** 2)
    kind = int(param) % 3
    if kind == 0:                                   # radial / vignette (off-centre)
        ox, oy = rng.uniform(-0.3, 0.3, 2)
        rr = np.sqrt(((xx - cx) / (W / 2) - ox) ** 2 + ((yy - cy) / (H / 2) - oy) ** 2)
        g = 1.0 - amp * np.clip(rr, 0, 1.5) ** 2
    elif kind == 1:                                 # low-frequency Fourier sum
        g = np.ones((H, W), np.float32)
        for _ in range(3):
            fx, fy = rng.uniform(0.5, 2.5, 2)
            ph = rng.uniform(0, 2 * np.pi)
            g += (amp / 3) * np.cos(2 * np.pi * (fx * xx / W + fy * yy / H) + ph)
    else:                                           # smooth random Gaussian-blurred field
        base = rng.standard_normal((H, W)).astype(np.float32)
        g = 1.0 + amp * cv2.GaussianBlur(base, (0, 0), sigmaX=min(H, W) / 6.0)
    g = g / max(1e-3, float(g.mean()))              # normalize to mean ~ 1
    return _apply_gain(bgr, g.astype(np.float32))


# ---------- F3 near-field physically-motivated relight (geometry-dependent) ----------
def _f3_physics(bgr, depth, param, seed, dataset="c3vd"):
    """Geometry-dependent relight: surface normals from the GT depth map (as a height field) drive a directional
    Lambertian shading; moving the light direction re-shades by SURFACE ORIENTATION, and a near-field inverse-square
    falloff re-weights by DISTANCE. gain = shading_new / shading_old applied to RGB (albedo x shading recomposition).
    Self-contained (no intrinsics) so it is robust across datasets; geometry (depth) is unchanged."""
    H, W = bgr.shape[:2]
    rng = np.random.default_rng(seed)
    D = depth.astype(np.float32)
    m = D > 0
    Dn = D / (np.nanmax(D) + 1e-6)
    Dn = cv2.GaussianBlur(Dn, (0, 0), sigmaX=max(H, W) / 300.0)     # denoise before gradients
    scl = 0.03                                                      # normal-tilt scale: small -> strong orientation sensitivity
    gx = cv2.Sobel(Dn, cv2.CV_32F, 1, 0, ksize=5) / scl
    gy = cv2.Sobel(Dn, cv2.CV_32F, 0, 1, ksize=5) / scl
    nz = np.ones_like(gx)
    nn = np.sqrt(gx * gx + gy * gy + 1.0) + 1e-6
    n = np.stack([-gx / nn, -gy / nn, nz / nn], axis=-1)           # HxWx3 surface normals
    a = 1.0 / (Dn + 0.15) ** 2; a = a / (np.nanmedian(a[m]) + 1e-9)  # near-field inverse-square (near=bright)
    def shade(l_dir, falloff):
        l = l_dir / (np.linalg.norm(l_dir) + 1e-9)
        ndl = np.clip(n[..., 0] * l[0] + n[..., 1] * l[1] + n[..., 2] * l[2], 0.05, None)
        return ndl * (a ** falloff)
    s_old = shade(np.array([0, 0, 1.0], np.float32), 0.0)          # frontal, no extra falloff (~ co-located)
    tilt = rng.uniform(-0.7, 0.7, 2).astype(np.float32)           # tilt the light -> orientation-dependent re-shade
    s_new = shade(np.array([tilt[0], tilt[1], 1.0], np.float32), float(rng.uniform(-0.5, 0.5)))
    ratio = (s_new + 0.05) / (s_old + 0.05)
    ratio = np.clip(ratio, 0.35, 3.0)
    ratio = np.where(m, ratio, 1.0)
    return _apply_gain(bgr, ratio.astype(np.float32))


# ---------- F4 / F5 depth-conditional luminance control ----------
def _depth_lum_trend(bgr, depth, mask, nbin=12):
    """Smooth depth->mean-luminance trend f(D) over valid pixels (monotone-ish, binned + smoothed)."""
    L = _lum(bgr); D = depth.astype(np.float32)
    v = mask & np.isfinite(D) & (D > 0)
    if v.sum() < 500:
        return None
    dq = np.quantile(D[v], np.linspace(0, 1, nbin + 1))
    dq[0] -= 1e-6; dq[-1] += 1e-6
    fD = np.full(bgr.shape[:2], np.nan, np.float32)
    binmean = np.zeros(nbin, np.float32)
    idx = np.clip(np.digitize(D, dq) - 1, 0, nbin - 1)
    for b in range(nbin):
        sel = v & (idx == b)
        binmean[b] = L[sel].mean() if sel.sum() > 0 else np.nan
    # fill + light smoothing of the bin means
    good = np.isfinite(binmean)
    binmean = np.interp(np.arange(nbin), np.where(good)[0], binmean[good])
    binmean = np.convolve(binmean, np.ones(3) / 3, mode="same")
    fD = binmean[idx]
    return L, D, v, fD, binmean, idx


def _f4_decorr(bgr, depth, param, seed, iters=4, nbin=24):
    """Equalize luminance across depth: iteratively divide out the depth->luminance trend so E[L|D] ~ const,
    driving |rho(L,D)| toward 0. Gain is lightly smoothed each pass for spatial plausibility."""
    m = _tissue_mask(bgr, depth)
    cur = bgr.copy()
    sig = max(bgr.shape[:2]) / 60.0
    for _ in range(iters):
        tr = _depth_lum_trend(cur, depth, m, nbin=nbin)
        if tr is None:
            break
        L, D, v, fD, binmean, idx = tr
        target = float(np.nanmean(binmean))
        gain = np.where(fD > 1e-3, target / fD, 1.0).astype(np.float32)
        gain = cv2.GaussianBlur(gain, (0, 0), sigmaX=sig)
        gain = np.where(m, gain, 1.0)
        cur = _apply_gain(cur, gain)
    return cur


def _f5_invert(bgr, depth, param, seed, k=0.6):
    """Decorrelate then inject a NEGATIVE brightness-vs-nearness trend (brighten far / darken near) -> cue conflict."""
    m = _tissue_mask(bgr, depth)
    tr = _depth_lum_trend(bgr, depth, m)
    if tr is None:
        return bgr.copy()
    L, D, v, fD, binmean, idx = tr
    target = float(np.nanmean(binmean))
    base_gain = np.where(fD > 1e-3, target / fD, 1.0).astype(np.float32)      # decorrelating gain
    # inject far-bright/near-dark: z-scored depth, far (large D) -> >1, near -> <1
    dz = (D - np.nanmean(D[v])) / (np.nanstd(D[v]) + 1e-6)
    inv_gain = 1.0 + k * np.clip(dz, -2.5, 2.5)
    gain = np.clip(base_gain * inv_gain, 0.3, 3.0).astype(np.float32)
    gain = cv2.GaussianBlur(gain, (0, 0), sigmaX=max(bgr.shape[:2]) / 40.0)
    gain = np.where(m, gain, 1.0)
    return _apply_gain(bgr, gain)


_DISPATCH = {"global": _f1_global, "spatial": _f2_spatial, "physics": _f3_physics,
             "decorr": _f4_decorr, "invert": _f5_invert}


def relight_dense(bgr, depth, family, param=0, seed=0, dataset="c3vd"):
    """bgr: HxWx3 uint8; depth: HxW float GT (larger=farther). Returns relit HxWx3 uint8, geometry unchanged."""
    if family == "physics":
        return _f3_physics(bgr, depth, param, seed, dataset=dataset)
    return _DISPATCH[family](bgr, depth, param, seed)


def realized_rho(bgr_relit, depth, mask=None):
    """Spearman-like correlation of realized luminance vs depth over valid pixels (audit F4/F5)."""
    from scipy.stats import spearmanr
    if mask is None:
        mask = (depth > 0) & (_lum(bgr_relit) > 3)
    L = _lum(bgr_relit)[mask]; D = depth[mask]
    if L.size < 100:
        return float("nan")
    # subsample for speed
    if L.size > 20000:
        i = np.random.default_rng(0).choice(L.size, 20000, replace=False); L, D = L[i], D[i]
    return float(spearmanr(L, D).correlation)
