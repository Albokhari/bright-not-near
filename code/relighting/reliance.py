"""§16 reliance.py — the spine metric: a per-model CAUSAL brightness-reliance 'swing'.

Render each image three ways and re-infer:
  • consistent   relight (brighten NEAR / darken FAR)  → reinforces correct geometry
  • adversarial  relight (brighten FAR  / darken NEAR)  → the trap
  • order_preserving null: a single GLOBAL gain (brighten-all vs darken-all) — changes brightness but NOT
    the per-point order, so a model that uses geometry (not the bright=near shortcut) must NOT swing.

  swing(model, img)  = pairwise_acc(consistent) − pairwise_acc(adversarial)     (>0 ⇒ relies on brightness)
  null(model, img)   = pairwise_acc(global-bright) − pairwise_acc(global-dark)   (≈0 if the operator is clean)
  reliance           = swing − null                                              (the causal effect of the
                       per-point brightness manipulation, beyond any global-brightness/clipping artifact)

pairwise_acc is over the GT ordered pairs (the correct depth order never changes; only the image is relit).
A confounded model (base DAV2) should show swing ≫ null; a shortcut-robust one (EndoOmni, §13-cured) ≈ 0.
"""
import numpy as np
import cv2

AMP = 0.5


def relit_consistent(bgr, xy, ranks, seed=0):
    import relight as RL
    return RL.relit(bgr, xy, "consistent", ranks=ranks, amp=AMP, seed=seed)


def relit_adversarial(bgr, xy, ranks, seed=0):
    import relight as RL
    return RL.relit(bgr, xy, "adversarial", ranks=ranks, amp=AMP, seed=seed)


def global_gain(bgr, g):
    """Uniform multiplicative gain (order_preserving null) in display space."""
    return np.clip(bgr.astype(np.float32) * g, 0, 255).astype(np.uint8)


def pairwise_acc(near, ranks):
    """Fraction of GT ordered pairs (i nearer than j) the prediction gets right (s[i] > s[j])."""
    s = np.asarray(near, float); r = np.asarray(ranks, float)
    c = t = 0
    for i in range(5):
        for j in range(5):
            if r[i] < r[j]:
                t += 1; c += int(s[i] > s[j])
    return c / t if t else np.nan


def discordant_acc(near, ranks, bright):
    """Pairwise accuracy restricted to brightness-adversarial pairs (nearer point darker) — for the clean guard."""
    s, r, b = map(lambda a: np.asarray(a, float), (near, ranks, bright))
    c = t = 0
    for i in range(5):
        for j in range(5):
            if r[i] < r[j] and b[i] < b[j]:
                t += 1; c += int(s[i] > s[j])
    return c / t if t else np.nan


def relit(bgr, xy, mode, ranks=None, amp=AMP, seed=0):
    import relight as RL
    return RL.relit(bgr, xy, mode, ranks=ranks, amp=amp, seed=seed)


def reliance_for_image(adapter, bgr, xy, ranks, ohw, fold=None, amp=AMP):
    """Per-image reliance with THREE nulls (audit-hardened):
      swing       = pairwise(consistent) − pairwise(adversarial)            [brightness-ORDER treatment]
      null_global = pairwise(global-bright) − pairwise(global-dark)         [global brightness magnitude]
      null_local  = pairwise(uniform-bump-bright) − pairwise(...-dark)      [S2: localized, SAME downsampling
                    as the treatment, order-preserving → controls resolution-dependent bump attenuation]
      null_scram  = pairwise(scramble seedA) − pairwise(scramble seedB)     [S3: same footprint + same gain
                    magnitudes as adversarial, RANDOM point assignment → isolates the brightness↔order link]
    reliance = swing − null_scram  (the strongest control). Also returns clean_disc guard."""
    sc = lambda im: adapter.score_relit(im, xy, ohw, fold=fold)
    s_clean = sc(bgr)
    a_clean = pairwise_acc(s_clean, ranks)
    a_cons = pairwise_acc(sc(relit(bgr, xy, "consistent", ranks)), ranks)
    a_adv = pairwise_acc(sc(relit(bgr, xy, "adversarial", ranks)), ranks)
    a_gb = pairwise_acc(sc(global_gain(bgr, 1.0 + amp)), ranks)
    a_gd = pairwise_acc(sc(global_gain(bgr, 1.0 - amp)), ranks)
    a_ub = pairwise_acc(sc(relit(bgr, xy, "uniform_bump", ranks, amp=+amp)), ranks)
    a_ud = pairwise_acc(sc(relit(bgr, xy, "uniform_bump", ranks, amp=-amp)), ranks)
    a_sa = pairwise_acc(sc(relit(bgr, xy, "scramble", ranks, seed=1)), ranks)
    a_sb = pairwise_acc(sc(relit(bgr, xy, "scramble", ranks, seed=2)), ranks)
    swing = a_cons - a_adv
    null_global = a_gb - a_gd
    null_local = a_ub - a_ud
    null_scram = a_sa - a_sb
    return dict(clean=a_clean, consistent=a_cons, adversarial=a_adv,
                null_global=null_global, null_local=null_local, null_scram=null_scram,
                swing=swing, reliance=swing - null_scram,
                clean_disc=discordant_acc(s_clean, ranks, _pt_bright(bgr, xy)))


def _pt_bright(bgr, xy):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(float)
    g = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    xy = np.asarray(xy)
    bx = np.clip(xy[:, 0].astype(int), 0, g.shape[1] - 1)
    by = np.clip(xy[:, 1].astype(int), 0, g.shape[0] - 1)
    return g[by, bx]
