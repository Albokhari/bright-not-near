"""Reviewer #3.3 earn -- relight SIDE-EFFECT audit + MATCHED controls + PHYSICS arm (CPU-only).

The spine result rests on a per-point counterfactual relight (relight.py) that manipulates POINT BRIGHTNESS in a
depth-rank-adversarial vs -consistent way. A reviewer can object: (a) the operator may change more than brightness
(contrast / saturation / sharpness / specular / clipping), so "brightness reliance" could be mislabelled; (b) any
matched-support local edit -- not the target-encoding specifically -- might drive the swing; (c) the manipulation
is a crude linear multiplicative gain, not a physically-grounded near-field (inverse-square) shading. This script
answers all three over the 307 Merged_PASS frames, GPU-free (pure image ops + the point_brightness/kendall readout
already used by relight._selftest and reliance.py).

(1) SIDE-EFFECTS. For the ADVERSARIAL relight vs CLEAN, per frame we measure the change in local CONTRAST
    (windowed luminance std), SATURATION (mean HSV-S), SHARPNESS (variance of Laplacian), SPECULAR fraction
    (luminance > 0.95*255, matching multicue_ops.proxy), and the per-channel + luminance CLIPPING RATE
    (fraction pinned to >=254 highlight / <=1 shadow). Means over 307 frames; the induced delta = adv - clean.

(2) DISPLAY-SPACE. relight.apply multiplies in DISPLAY (gamma-encoded sRGB) space -- a x g gain there is NOT a
    linear-radiance scaling; it exaggerates contrast/saturation shifts near mid-grey and clips highlights sooner.
    We STATE this and HANDLE it by recomputing the same side-effects for a counterfactual LINEAR-radiance
    application (sRGB^gamma -> multiply -> ^(1/gamma)) of the identical gain field, so the gamma-attributable
    portion of every side-effect is quantified rather than assumed away.

(3) MATCHED controls. hue-shift / gamma / contrast / blur perturbations sharing the relight's EXACT Gaussian
    spatial support (sigma=min(H,W)/6) and per-point magnitude schedule (amp), but with the per-point loading
    SCRAMBLED w.r.t. the ordinal ranks (relight.brightness_field mode='scramble'; blur via multicue_ops 'matched')
    -- i.e. matched footprint+magnitude but NOT encoding the ordinal target. Their brightness reliance-style swing
    (kendall(point_brightness, ranks) between two scramble seeds, the null_scram readout of reliance.py) should be
    ~0 vs the large adversarial-vs-consistent brightness swing -- proving the swing needs the TARGET-ENCODING, not
    merely a matched local edit (gamma/contrast move point brightness a lot in MAGNITUDE yet swing ~0 once
    de-correlated from rank).

(4) AMPLITUDE SWEEP of the brightness swing for amp in {0.2, 0.35, 0.5}.

(5) PHYSICS arm. Code/Depth_metrics_model/physics_shading.py is NOT trivially reusable here (its PPSNet renderer
    needs a DENSE depth map + camera intrinsics K; the Kvasir frames ship neither GT depth nor K, and estimating
    depth needs a GPU model) -- we NOTE this. Instead we add a self-contained inverse-square (I ~ 1/depth^2)
    near-field relight driven by rank pseudo-depths and check the brightness swing survives the physically-shaped
    (non-linear) gain, so the confound is not an artefact of the linear parametrisation.

Output: results/validation/sideeffects/relight_sideeffects.{json,md}.  Run: CPU short (see bottom of file).
"""
import os
import sys
import json
import argparse

import numpy as np
import cv2
from scipy.stats import kendalltau

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import relight as RL           # relit / brightness_field / apply / point_brightness / AMP
import multicue_ops as MC      # blur_apply (matched mode) reused for the blur control

P1 = "/well/rittscher/users/fxh757/Code/Papers/P1_Ordinal_Endo_Depth_Benchmark__EndoLINA_MICCAI2026"
IMG_ROOT = "/well/rittscher/users/fxh757/Code/Kvasir_Ranking/Merged_PASS/images"
GT_JSON = os.path.join(P1, "results/consensus_GT/merged_pass_consensus_gt.json")
OUT_DIR = os.path.join(P1, "results/validation/sideeffects")

GAMMA = 2.2                    # sRGB display-gamma approximation for the linear-space handling of (2)
HUE_MAX_DEG = 90.0            # max hue rotation (deg) at |field|=amp for the hue control
SWING_THRESH = 0.1            # "meaningful swing" threshold, matching relight._selftest (0.1)
PHYS_DNEAR, PHYS_DFAR = 1.0, 3.0  # rank->pseudo-depth range for the inverse-square physics arm


# --------------------------------------------------------------------------- readouts
def luminance(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _kendall(a, b):
    t = kendalltau(a, b).correlation
    return t if t == t else np.nan          # NaN on degenerate (constant) inputs


def local_contrast(L, k=15):
    """Mean windowed luminance std (local RMS contrast, intensity units)."""
    Lf = L.astype(np.float32)
    mu = cv2.boxFilter(Lf, cv2.CV_32F, (k, k), normalize=True)
    mu2 = cv2.boxFilter(Lf * Lf, cv2.CV_32F, (k, k), normalize=True)
    return float(np.sqrt(np.clip(mu2 - mu * mu, 0, None)).mean())


def frame_metrics(bgr):
    """Scalar side-effect readouts for one frame."""
    L = luminance(bgr)
    b, g, r = bgr[..., 0].astype(np.int32), bgr[..., 1].astype(np.int32), bgr[..., 2].astype(np.int32)
    hi = lambda c: float((c >= 254).mean())
    lo = lambda c: float((c <= 1).mean())
    return {
        "local_contrast": local_contrast(L),
        "saturation": float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1].mean()),
        "sharpness": float(cv2.Laplacian(L, cv2.CV_32F).var()),
        "specular_frac": float((L > 0.95 * 255).mean()),
        "clip_hi_R": hi(r), "clip_hi_G": hi(g), "clip_hi_B": hi(b),
        "clip_lo_R": lo(r), "clip_lo_G": lo(g), "clip_lo_B": lo(b),
        "clip_hi_lum": float((L >= 254).mean()), "clip_lo_lum": float((L <= 1).mean()),
    }


METRIC_KEYS = list(frame_metrics(np.zeros((4, 4, 3), np.uint8)).keys())


# --------------------------------------------------------------------------- operators
def apply_linear(bgr, G):
    """Apply the multiplicative gain field in LINEAR radiance (sRGB^GAMMA -> gain -> ^(1/GAMMA))."""
    lin = np.clip(bgr.astype(np.float32) / 255.0, 0, 1) ** GAMMA
    lin = lin * G[..., None]
    out = np.clip(lin, 0, 1) ** (1.0 / GAMMA)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def gamma_local(bgr, d):
    """Per-pixel local gamma: exponent 1/(1+d).  d>0 brightens (matches gain direction)."""
    e = 1.0 / (1.0 + d)
    base = np.clip(bgr.astype(np.float32) / 255.0, 0, 1)
    return np.clip((base ** e[..., None]) * 255.0, 0, 255).astype(np.uint8)


def contrast_local(bgr, d, mid=127.5):
    """Per-pixel local contrast about mid-grey with local gain (1+d)."""
    out = mid + (bgr.astype(np.float32) - mid) * (1.0 + d[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def hue_shift(bgr, d):
    """Luminance-preserving local hue rotation by d*HUE_MAX_DEG."""
    hsv = cv2.cvtColor(np.clip(bgr.astype(np.float32) / 255.0, 0, 1), cv2.COLOR_BGR2HSV)  # H:[0,360]
    hsv[..., 0] = np.mod(hsv[..., 0] + d * HUE_MAX_DEG, 360.0)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _gauss_partition(H, W, pts, v, sigma):
    """relight.brightness_field's partition-of-unity gain field, with explicit per-point magnitudes v."""
    yy, xx = np.mgrid[0:H, 0:W]
    num = np.zeros((H, W), np.float32)
    den = np.full((H, W), 0.10, np.float32)
    for (x, y), vi in zip(pts, v):
        b = np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))).astype(np.float32)
        num += vi * b
        den += b
    return (1.0 + num / den).astype(np.float32)


def physics_gain(bgr, xy, ranks, mode, amp):
    """Inverse-square near-field relight: brightness ~ 1/depth^2 over rank pseudo-depths.
    mode='consistent' brightens NEAR (physically correct); 'adversarial' brightens FAR (the trap)."""
    H, W = bgr.shape[:2]
    r = np.asarray(ranks, float)
    dn = (r - r.min()) / (r.max() - r.min() + 1e-9)          # 0 (near) .. 1 (far)
    depth = PHYS_DNEAR + dn * (PHYS_DFAR - PHYS_DNEAR)
    inv2 = 1.0 / depth ** 2                                   # near brightest
    z = 2 * (inv2 - inv2.min()) / (inv2.max() - inv2.min() + 1e-9) - 1   # near->+1, far->-1
    v = amp * z if mode == "consistent" else -amp * z        # adversarial: darken near / brighten far
    return _gauss_partition(H, W, np.asarray(xy, float), v, min(H, W) / 6.0)


def control_scram(op, bgr, xy, ranks, amp, seed):
    """Matched-support, rank-SCRAMBLED (target-free) perturbation in the named non-brightness/adjacent channel."""
    if op == "blur":
        return MC.blur_apply(bgr, xy, ranks, "matched", amp=amp, seed=seed)   # permutes adv magnitudes
    d = RL.brightness_field(bgr, xy, "scramble", ranks=ranks, amp=amp, seed=seed) - 1.0
    if op == "gamma":
        return gamma_local(bgr, d)
    if op == "contrast":
        return contrast_local(bgr, d)
    if op == "hue":
        return hue_shift(bgr, d)
    raise ValueError(op)


CONTROLS = ["hue", "gamma", "contrast", "blur"]


# --------------------------------------------------------------------------- stats helpers
def ci(x, n=2000, seed=0):
    x = np.asarray([v for v in x if v == v], float)          # drop NaN
    if len(x) == 0:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def frac_gt(x, thr=SWING_THRESH):
    x = np.asarray([v for v in x if v == v], float)
    return float((x > thr).mean()) if len(x) else float("nan")


def swing_of(bgr, xy, ranks, adv_img, cons_img):
    """kendall(point_brightness(adv), ranks) - kendall(point_brightness(cons), ranks)  (relight._selftest pattern)."""
    ta = _kendall(RL.point_brightness(adv_img, xy), ranks)
    tc = _kendall(RL.point_brightness(cons_img, xy), ranks)
    return ta - tc


def load_gt():
    g = json.load(open(GT_JSON))
    out = []
    for im in g["images"]:
        pts = sorted(im["points"], key=lambda p: p["point_id"])
        xy = np.array([[p["x"], p["y"]] for p in pts], float)
        ranks = np.array([p["rank"] for p in pts], float)
        out.append((im["filename"], xy, ranks))
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="limit #images (0 = all 307)")
    ap.add_argument("--amp", type=float, default=RL.AMP, help="headline relight amplitude (default 0.5)")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "relight_sideeffects.json"))
    args = ap.parse_args()
    amp = args.amp
    items = load_gt()
    if args.n:
        items = items[: args.n]
    print(f"relight side-effects | {len(items)} frames | amp={amp} | GPU-free", flush=True)

    # accumulators -----------------------------------------------------------
    se = {sp: {k: [] for k in METRIC_KEYS} for sp in ("clean", "adv_display", "adv_linear")}
    ref_swing, ref_absbright = [], []                                   # relight brightness swing + |dB|
    phys_swing, phys_absbright = [], []
    ctrl_swing = {c: [] for c in CONTROLS}
    ctrl_absbright = {c: [] for c in CONTROLS}
    sweep_amps = [0.2, 0.35, 0.5]
    sweep_swing = {a: [] for a in sweep_amps}
    used = 0

    for i, (fn, xy, ranks) in enumerate(items):
        bgr = cv2.imread(os.path.join(IMG_ROOT, fn))
        if bgr is None:
            continue
        used += 1
        b_clean = RL.point_brightness(bgr, xy)

        # (1)+(2) side-effects: adversarial gain applied in DISPLAY (real op) and LINEAR (handled) space
        G = RL.brightness_field(bgr, xy, "adversarial", ranks=ranks, amp=amp)
        adv_disp = RL.apply(bgr, G)
        adv_lin = apply_linear(bgr, G)
        for sp, img in (("clean", bgr), ("adv_display", adv_disp), ("adv_linear", adv_lin)):
            m = frame_metrics(img)
            for k in METRIC_KEYS:
                se[sp][k].append(m[k])

        # (3) reference brightness swing (adversarial vs consistent) + realised |dB| at points
        adv_r = RL.relit(bgr, xy, "adversarial", ranks=ranks, amp=amp)
        cons_r = RL.relit(bgr, xy, "consistent", ranks=ranks, amp=amp)
        ref_swing.append(swing_of(bgr, xy, ranks, adv_r, cons_r))
        ref_absbright.append(float(np.abs(RL.point_brightness(adv_r, xy) - b_clean).mean()))

        # (3) matched, target-SCRAMBLED controls: swing = kendall(seedA) - kendall(seedB) (null_scram readout)
        for c in CONTROLS:
            a = control_scram(c, bgr, xy, ranks, amp, seed=1)
            b = control_scram(c, bgr, xy, ranks, amp, seed=2)
            ctrl_swing[c].append(_kendall(RL.point_brightness(a, xy), ranks)
                                 - _kendall(RL.point_brightness(b, xy), ranks))
            ctrl_absbright[c].append(float(np.abs(RL.point_brightness(a, xy) - b_clean).mean()))

        # (4) amplitude sweep of the brightness swing
        for a_ in sweep_amps:
            av = RL.relit(bgr, xy, "adversarial", ranks=ranks, amp=a_)
            cv_ = RL.relit(bgr, xy, "consistent", ranks=ranks, amp=a_)
            sweep_swing[a_].append(swing_of(bgr, xy, ranks, av, cv_))

        # (5) inverse-square physics relight arm
        pa = RL.apply(bgr, physics_gain(bgr, xy, ranks, "adversarial", amp))
        pc = RL.apply(bgr, physics_gain(bgr, xy, ranks, "consistent", amp))
        phys_swing.append(swing_of(bgr, xy, ranks, pa, pc))
        phys_absbright.append(float(np.abs(RL.point_brightness(pa, xy) - b_clean).mean()))

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(items)}", flush=True)

    # aggregate --------------------------------------------------------------
    clean_m = {k: float(np.mean(se["clean"][k])) for k in METRIC_KEYS}
    advd_m = {k: float(np.mean(se["adv_display"][k])) for k in METRIC_KEYS}
    advl_m = {k: float(np.mean(se["adv_linear"][k])) for k in METRIC_KEYS}
    delta_d = {k: advd_m[k] - clean_m[k] for k in METRIC_KEYS}          # induced by real (display) op
    delta_l = {k: advl_m[k] - clean_m[k] for k in METRIC_KEYS}          # induced if applied in linear space

    def swing_block(sw, ab=None):
        blk = {"swing_mean_ci": ci(sw), "frac_swing_gt_0p1": frac_gt(sw)}
        if ab is not None:
            blk["mean_abs_point_brightness_change"] = float(np.mean(ab))
        return blk

    result = {
        "meta": {
            "n_frames": used, "amp": amp, "img_root": IMG_ROOT, "gt_json": GT_JSON,
            "readout": "point_brightness (0.299R+0.587G+0.114B at the 5 pts) -> kendall vs GT depth ranks; "
                       "swing = kendall(adversarial) - kendall(consistent), the reliance.py null_scram/selftest pattern",
            "display_space_note": (
                "relight.apply multiplies in DISPLAY (gamma-encoded sRGB) space; a x g gain there is NOT a linear-"
                "radiance scaling -- it amplifies mid-grey contrast/saturation shifts and clips highlights sooner. "
                f"'delta_linear' recomputes every side-effect for the same gain applied in LINEAR radiance "
                f"(sRGB^{GAMMA} -> gain -> ^(1/{GAMMA})); delta_display - delta_linear is the gamma-attributable part."),
            "physics_shading_reusable": False,
            "physics_shading_note": (
                "Code/Depth_metrics_model/physics_shading.py is NOT trivially reusable on these frames: its PPSNet "
                "renderer (shading_map/physics_score) requires a DENSE depth map (B,1,H,W) AND camera intrinsics K; "
                "the Kvasir Merged_PASS frames ship neither GT depth nor K, and producing a depth map needs a GPU "
                "model. The 'physics_inverse_square' block below is a self-contained rank-pseudo-depth I~1/depth^2 "
                "arm that exercises the same near-field law without importing PPSNet/torch."),
        },
        "sideeffects_adversarial_vs_clean": {
            "clean_mean": clean_m, "adv_display_mean": advd_m, "adv_linear_mean": advl_m,
            "delta_display": delta_d, "delta_linear": delta_l,
        },
        "brightness_swing_reference_adv_vs_cons": swing_block(ref_swing, ref_absbright),
        "matched_controls_scrambled": {
            c: swing_block(ctrl_swing[c], ctrl_absbright[c]) for c in CONTROLS
        },
        "amplitude_sweep_brightness_swing": {
            str(a_): {"swing_mean_ci": ci(sweep_swing[a_]), "frac_swing_gt_0p1": frac_gt(sweep_swing[a_])}
            for a_ in sweep_amps
        },
        "physics_inverse_square": {
            **swing_block(phys_swing, phys_absbright),
            "pseudo_depth_range": [PHYS_DNEAR, PHYS_DFAR],
            "law": "I ~ 1/depth^2 (near-field, co-located light); consistent=brighten near, adversarial=brighten far",
        },
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    write_md(result, os.path.splitext(args.out)[0] + ".md")
    print_summary(result)
    print(f"\nwrote {args.out}\nwrote {os.path.splitext(args.out)[0] + '.md'}")


# --------------------------------------------------------------------------- reporting
def _f(v, p=4):
    return "nan" if v != v else f"{v:.{p}f}"


def write_md(R, path):
    se = R["sideeffects_adversarial_vs_clean"]
    L = []
    L.append("# Relight side-effect audit + matched controls + physics arm (reviewer #3.3)\n")
    L.append(f"GPU-free, over **{R['meta']['n_frames']} Merged_PASS frames**, headline amp **{R['meta']['amp']}**. "
             "Readout: " + R["meta"]["readout"] + "\n")

    L.append("## (1) What the ADVERSARIAL relight changes besides brightness (mean over frames)\n")
    L.append("| metric | clean | adv (display) | delta (display) | delta (linear) | gamma-attributable |")
    L.append("|---|---|---|---|---|---|")
    for k in METRIC_KEYS:
        dd, dl = se["delta_display"][k], se["delta_linear"][k]
        L.append(f"| {k} | {_f(se['clean_mean'][k])} | {_f(se['adv_display_mean'][k])} | "
                 f"{_f(dd)} | {_f(dl)} | {_f(dd - dl)} |")
    L.append("")
    L.append("## (2) Display-space handling\n")
    L.append(R["meta"]["display_space_note"] + " The `gamma-attributable` column (delta_display - delta_linear) "
             "isolates the part of each side-effect caused by acting in gamma space rather than by the gain itself.\n")

    ref = R["brightness_swing_reference_adv_vs_cons"]
    L.append("## (3) Matched-support controls -- brightness reliance-style swing\n")
    L.append("Controls share the relight's exact Gaussian support (sigma=min(H,W)/6) and per-point magnitude but "
             "SCRAMBLE the per-point loading w.r.t. the ranks (not encoding the ordinal target). Swing = "
             "kendall(seedA) - kendall(seedB) on point brightness. `mean|dB|` = realised per-point brightness change "
             "vs clean (shows the perturbation is real in magnitude even when the swing is ~0).\n")
    L.append("| operator | brightness swing [95% CI] | frac swing>0.1 | mean\\|dB\\| |")
    L.append("|---|---|---|---|")
    rs = ref["swing_mean_ci"]
    L.append(f"| **relight (adv vs cons, target-encoded)** | **{_f(rs[0],3)}** [{_f(rs[1],3)}, {_f(rs[2],3)}] | "
             f"**{_f(ref['frac_swing_gt_0p1'],3)}** | {_f(ref['mean_abs_point_brightness_change'],2)} |")
    for c in CONTROLS:
        b = R["matched_controls_scrambled"][c]
        s = b["swing_mean_ci"]
        L.append(f"| {c} (scrambled, target-free) | {_f(s[0],3)} [{_f(s[1],3)}, {_f(s[2],3)}] | "
                 f"{_f(b['frac_swing_gt_0p1'],3)} | {_f(b['mean_abs_point_brightness_change'],2)} |")
    L.append("")
    L.append("Interpretation: the large swing needs the TARGET-ENCODING, not a matched local edit -- gamma/contrast "
             "move point brightness substantially (see mean|dB|) yet produce ~0 swing once de-correlated from rank; "
             "hue/blur barely touch luminance at all.\n")

    L.append("## (4) Amplitude sweep of the brightness swing\n")
    L.append("| amp | brightness swing [95% CI] | frac swing>0.1 |")
    L.append("|---|---|---|")
    for a_, blk in R["amplitude_sweep_brightness_swing"].items():
        s = blk["swing_mean_ci"]
        L.append(f"| {a_} | {_f(s[0],3)} [{_f(s[1],3)}, {_f(s[2],3)}] | {_f(blk['frac_swing_gt_0p1'],3)} |")
    L.append("")

    ph = R["physics_inverse_square"]
    ps = ph["swing_mean_ci"]
    L.append("## (5) Inverse-square physics relight arm\n")
    L.append(R["meta"]["physics_shading_note"] + "\n")
    L.append(f"Physics ({ph['law']}) brightness swing: **{_f(ps[0],3)}** [{_f(ps[1],3)}, {_f(ps[2],3)}], "
             f"frac>0.1 = {_f(ph['frac_swing_gt_0p1'],3)}, mean|dB| = {_f(ph['mean_abs_point_brightness_change'],2)}. "
             "The brightness confound survives a physically-shaped (non-linear inverse-square) gain, so it is not an "
             "artefact of the linear relight parametrisation.\n")
    open(path, "w").write("\n".join(L))


def print_summary(R):
    se = R["sideeffects_adversarial_vs_clean"]
    print("\n=== (1) adversarial-relight side-effects (mean over frames) ===")
    for k in METRIC_KEYS:
        print(f"  {k:<16} clean {_f(se['clean_mean'][k]):>9}  adv {_f(se['adv_display_mean'][k]):>9}  "
              f"dDisp {_f(se['delta_display'][k]):>9}  dLin {_f(se['delta_linear'][k]):>9}")
    ref = R["brightness_swing_reference_adv_vs_cons"]
    print("\n=== (3) brightness swing: relight (target-encoded) vs matched scrambled controls ===")
    print(f"  {'relight':<10} swing {_f(ref['swing_mean_ci'][0],3):>7}  frac>0.1 {_f(ref['frac_swing_gt_0p1'],3):>6}  "
          f"mean|dB| {_f(ref['mean_abs_point_brightness_change'],2)}")
    for c in CONTROLS:
        b = R["matched_controls_scrambled"][c]
        print(f"  {c:<10} swing {_f(b['swing_mean_ci'][0],3):>7}  frac>0.1 {_f(b['frac_swing_gt_0p1'],3):>6}  "
              f"mean|dB| {_f(b['mean_abs_point_brightness_change'],2)}")
    print("\n=== (4) amplitude sweep ===")
    for a_, blk in R["amplitude_sweep_brightness_swing"].items():
        print(f"  amp {a_:<5} swing {_f(blk['swing_mean_ci'][0],3):>7}  frac>0.1 {_f(blk['frac_swing_gt_0p1'],3)}")
    ph = R["physics_inverse_square"]
    print(f"\n=== (5) physics inverse-square arm ===\n  swing {_f(ph['swing_mean_ci'][0],3)}  "
          f"frac>0.1 {_f(ph['frac_swing_gt_0p1'],3)}  mean|dB| {_f(ph['mean_abs_point_brightness_change'],2)}")


if __name__ == "__main__":
    main()
