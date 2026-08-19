"""§16 reviewer #3.4 — does the relight-consistency CURE generalize to an INDEPENDENT relight operator?

Motivation (reviewer #3.4). The relight-consistency cure (results/finetune/dav2cons_fold*, trained by
finetune_dav2_consistency.py) was fit with brightness-invariance views drawn from relight.py mode='random'
(smooth Gaussian partition-of-unity bumps, per-point gains ~U[-amp,+amp]). The in-paper reliance diagnostic
(reliance.py / e1_reliance.py) measures the swing under mode='consistent'/'adversarial', which share that SAME
field family (rank-LINEAR gains, same sigma, same amp). A skeptic could argue the cure's reliance drop is
AUGMENTATION-MEMORIZATION: robustness to the very operator family it trained on. This probe re-measures the
reliance under a HELD-OUT operator family whose *gain law* the cure NEVER saw, and asks whether the cure's
reliance reduction TRANSFERS. Transfer ⇒ the cure learned genuine brightness-magnitude invariance (not a
memorized augmentation pattern).

HELD-OUT OPERATOR FAMILY (reuses relight.py modes 'scramble', 'uniform_bump', 'order_preserving' — the three
modes NOT used to fit the cure, and never used as the E1 *treatment*). Same 5 points, same sigma, same amp as
E1 (so amplitude is matched and the ONLY thing that changes is the operator's gain law → a fair reliance
comparison). Two held-out order-informative TREATMENTS + two held-out NULLS:

  • T_step  (from 'uniform_bump'): HARD rank-threshold bumps — near half darkened / far half brightened by a
              UNIFORM ±amp (a step gain law; 'uniform_bump' = uniform-magnitude localized bumps). This is the
              held-out analogue of consistent/adversarial but with a step law, not the E1/random continuous law.
  • T_swap  (from 'scramble'): the adversarial rank-linear MAGNITUDES are randomly permuted across points
              ('scramble'), but the SIGN follows the depth order (near darken / far brighten). Order-informative
              yet with magnitudes decorrelated from rank → a second, structurally distinct held-out treatment.
  • N_scram (pure 'scramble', two seeds): random point↔gain assignment → order-uninformative NULL (≈0).
  • N_op    ('order_preserving', two seeds): a single GLOBAL gain → order-uninformative NULL (≈0).

Per held-out operator, reliance = swing − null_scram   (identical form to reliance.py's reliance = swing − null).
Headline held-out reliance = mean(reliance_step, reliance_swap) − 0  (the two held-out treatments; each already
null-subtracted). Heads compared (5-fold OUT-OF-FOLD): base (no FT), ordinal-FT (dav2 fold heads),
relight-cure (dav2cons fold heads), all loaded via pool_inference/finetune_dav2. Image-clustered bootstrap CIs
(p1_common.cluster_bootstrap_ci — the reliance pipeline's bootstrap) per head and on PAIRED per-image reliance
differences (base/ordinal-FT/cure), Holm-corrected. TRANSFER verdict keyed on whether the cure's in-family
reliance reduction (significant vs ordinal-FT in E1) survives under the held-out operator.

Run (GPU — model forwards on relit images):
  module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1 Pillow/10.0.0-GCCcore-12.3.0
  export XFORMERS_DISABLED=1
  cd .../code/counterfactual && python3 cure_heldout_operator.py           # all images, 3 heads, 5-fold OOF
  python3 cure_heldout_operator.py --n 6                                     # quick pipeline check
"""
import os, sys, json, argparse
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                    # relight, reliance, pool_inference
sys.path.insert(0, os.path.join(HERE, "..", "validation"))  # p1_common
sys.path.insert(0, os.path.join(HERE, "..", "finetune"))    # finetune_dav2
import pool_inference as PI          # noqa: E402  DAVAdapter, load_dav2_fold_heads, fold_of, DEV
import reliance as RZ                # noqa: E402  pairwise_acc, discordant_acc, _pt_bright
import relight as RL                 # noqa: E402  relit, apply, AMP  (held-out modes: scramble/order_preserving)
import finetune_dav2 as FD           # noqa: E402  GTD, FOLDS, IMG_ROOT, P1
import p1_common as C                # noqa: E402  cluster_bootstrap_ci, holm

P1 = FD.P1
IMG_ROOT = FD.IMG_ROOT
GTD = FD.GTD
FOLDS = FD.FOLDS


# ------------------------------- held-out operator family -------------------------------
def _rn(ranks):
    """rank → [-1,1] (near rank1 = -1, far = +1) — same normalisation as relight.brightness_field."""
    r = np.asarray(ranks, float)
    return 2 * (r - r.min()) / (r.max() - r.min() + 1e-9) - 1


def _field_from_v(bgr, pts, v):
    """Smooth multiplicative gain field from per-point gains v — IDENTICAL Gaussian partition-of-unity
    (sigma=min(H,W)/6, den-floor 0.10) as relight.brightness_field; only the gain vector v is held-out."""
    H, W = bgr.shape[:2]
    pts = np.asarray(pts, float)
    sigma = min(H, W) / 6.0
    yy, xx = np.mgrid[0:H, 0:W]
    num = np.zeros((H, W), np.float32)
    den = np.full((H, W), 0.10, np.float32)
    for (x, y), vi in zip(pts, v):
        b = np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))).astype(np.float32)
        num += float(vi) * b
        den += b
    return (1.0 + num / den).astype(np.float32)


def _apply_v(bgr, pts, v):
    return RL.apply(bgr, _field_from_v(bgr, pts, v))          # reuse relight's display-space clip


def _halfsign(ranks):
    """Median split: near half -> -1 (darken), far half -> +1 (brighten), median point -> 0 (neutral).
    Robust to float epsilon (unlike sign(_rn)); with 5 distinct ranks 1..5 this is a clean 2-0-2 split."""
    r = np.asarray(ranks, float)
    return np.sign(r - np.median(r))


def relit_step(bgr, xy, ranks, direction, amp):
    """HELD-OUT 'uniform_bump'-derived HARD step: far half +amp (brighten), near half -amp (darken),
    median neutral — UNIFORM magnitude (uniform_bump signature), a step gain law the cure never saw.
    direction='adversarial' -> brighten far / darken near (contradicts geometry); 'consistent' -> reverse."""
    base = amp * _halfsign(ranks)
    v = base if direction == "adversarial" else -base
    return _apply_v(bgr, xy, v)


def relit_swap(bgr, xy, ranks, direction, amp, seed):
    """HELD-OUT 'scramble'-derived directed operator: adversarial rank-linear MAGNITUDES permuted across
    points (scramble), SIGN follows the near/far depth order -> order-informative but magnitude-decorrelated
    from rank (a second, structurally distinct held-out gain law)."""
    mag = np.random.default_rng(seed).permutation(np.abs(amp * _rn(ranks)))
    base = _halfsign(ranks) * mag
    v = base if direction == "adversarial" else -base
    return _apply_v(bgr, xy, v)


def heldout_reliance_for_image(adapter, bgr, xy, ranks, ohw, fold, amp):
    """Per-image HELD-OUT reliance (all pairwise accuracies over the fixed GT depth order)."""
    sc = lambda im: adapter.score_relit(im, xy, ohw, fold=fold)
    s_clean = sc(bgr)
    a_clean = RZ.pairwise_acc(s_clean, ranks)
    # held-out treatment 1 — 'uniform_bump' hard step
    a_cons_st = RZ.pairwise_acc(sc(relit_step(bgr, xy, ranks, "consistent", amp)), ranks)
    a_adv_st = RZ.pairwise_acc(sc(relit_step(bgr, xy, ranks, "adversarial", amp)), ranks)
    # held-out treatment 2 — 'scramble'-directed
    a_cons_sw = RZ.pairwise_acc(sc(relit_swap(bgr, xy, ranks, "consistent", amp, seed=11)), ranks)
    a_adv_sw = RZ.pairwise_acc(sc(relit_swap(bgr, xy, ranks, "adversarial", amp, seed=11)), ranks)
    # held-out null 1 — pure 'scramble' (random point↔gain), two seeds
    a_sa = RZ.pairwise_acc(sc(RL.relit(bgr, xy, "scramble", ranks, amp=amp, seed=1)), ranks)
    a_sb = RZ.pairwise_acc(sc(RL.relit(bgr, xy, "scramble", ranks, amp=amp, seed=2)), ranks)
    # held-out null 2 — 'order_preserving' (global single gain), two seeds
    a_oa = RZ.pairwise_acc(sc(RL.relit(bgr, xy, "order_preserving", ranks, amp=amp, seed=1)), ranks)
    a_ob = RZ.pairwise_acc(sc(RL.relit(bgr, xy, "order_preserving", ranks, amp=amp, seed=2)), ranks)
    swing_step = a_cons_st - a_adv_st
    swing_swap = a_cons_sw - a_adv_sw
    null_scram = a_sa - a_sb
    null_op = a_oa - a_ob
    rel_step = swing_step - null_scram
    rel_swap = swing_swap - null_scram
    return dict(clean=a_clean, clean_disc=RZ.discordant_acc(s_clean, ranks, RZ._pt_bright(bgr, xy)),
                cons_step=a_cons_st, adv_step=a_adv_st, cons_swap=a_cons_sw, adv_swap=a_adv_sw,
                swing_step=swing_step, swing_swap=swing_swap, null_scram=null_scram, null_op=null_op,
                reliance_step=rel_step, reliance_swap=rel_swap, reliance_combined=0.5 * (rel_step + rel_swap))


# ------------------------------- driver -------------------------------
def build_heads():
    return {
        "base":        PI.DAVAdapter("base", fold_heads=None),
        "ordinalFT":   PI.DAVAdapter("ordinalFT", fold_heads=PI.load_dav2_fold_heads("dav2")),
        "relightCure": PI.DAVAdapter("relightCure", fold_heads=PI.load_dav2_fold_heads("dav2cons")),
    }


def item(fn):
    bgr = cv2.imread(os.path.join(IMG_ROOT, fn))
    im = GTD[fn]
    pts = sorted(im["points"], key=lambda p: p["point_id"])
    xy = np.array([[p["x"], p["y"]] for p in pts], float)
    ranks = np.array([p["rank"] for p in pts], float)
    return bgr, xy, ranks, (bgr.shape[0], bgr.shape[1])


KEYS = ("clean", "clean_disc", "cons_step", "adv_step", "cons_swap", "adv_swap",
        "swing_step", "swing_swap", "null_scram", "null_op",
        "reliance_step", "reliance_swap", "reliance_combined")


def _boot_p_le0(v):
    """One-sided bootstrap p(mean ≤ 0) over image clusters (matches e1_reliance)."""
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan
    rng = np.random.default_rng(0)
    bs = v[rng.integers(0, len(v), (10000, len(v)))].mean(1)
    return float((bs <= 0).mean())


def _paired(a_vec, b_vec):
    """Paired image-clustered diff a−b: CI + two-sided bootstrap p (matches e1_reliance)."""
    diff = a_vec - b_vec
    dm, dlo, dhi = C.cluster_bootstrap_ci(diff)
    dd = diff[~np.isnan(diff)]
    rng = np.random.default_rng(0)
    bs = dd[rng.integers(0, len(dd), (10000, len(dd)))].mean(1)
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return [dm, dlo, dhi], float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="limit #images (0 = all)")
    ap.add_argument("--amp", type=float, default=RL.AMP, help="held-out amplitude (default = E1 AMP for fairness)")
    ap.add_argument("--out", default="results/counterfactual/cure_heldout_operator")
    args = ap.parse_args()

    heads = build_heads()
    files = sorted(GTD.keys())
    if args.n:
        files = files[: args.n]
    print(f"[cure_heldout_operator] {len(files)} imgs | heads {list(heads)} | amp {args.amp} | dev {PI.DEV}",
          flush=True)

    per = {m: {k: [] for k in KEYS} for m in heads}
    used = []
    for n, fn in enumerate(files):
        bgr, xy, ranks, ohw = item(fn)
        if bgr is None:
            continue
        used.append(fn)
        oof = PI.fold_of(FOLDS, fn)                 # out-of-fold test head for this image (None for base)
        for m, ad in heads.items():
            f = None if m == "base" else oof        # base = single frozen head; FT heads = OOF per image
            r = heldout_reliance_for_image(ad, bgr, xy, ranks, ohw, f, args.amp)
            for k in per[m]:
                per[m][k].append(r[k])
        if (n + 1) % 25 == 0:
            print(f"  {n+1}/{len(files)}", flush=True)

    A = {m: {k: np.array(v, float) for k, v in per[m].items()} for m in heads}
    ci = lambda x: list(C.cluster_bootstrap_ci(x))

    # ---- per-head aggregates + image-clustered CIs + within-head p(reliance≤0) (Holm) ----
    res = {}; pvec = []
    for m in heads:
        d = A[m]
        p = _boot_p_le0(d["reliance_combined"]); pvec.append(p)
        res[m] = dict(
            clean_pairwise=float(np.nanmean(d["clean"])),
            clean_discordant=float(np.nanmean(d["clean_disc"])),
            cons_step=float(np.nanmean(d["cons_step"])), adv_step=float(np.nanmean(d["adv_step"])),
            cons_swap=float(np.nanmean(d["cons_swap"])), adv_swap=float(np.nanmean(d["adv_swap"])),
            swing_step=ci(d["swing_step"]), swing_swap=ci(d["swing_swap"]),
            null_scram=ci(d["null_scram"]), null_op=ci(d["null_op"]),
            reliance_step=ci(d["reliance_step"]), reliance_swap=ci(d["reliance_swap"]),
            reliance_combined=ci(d["reliance_combined"]), p_reliance_le0=p)
    for m, pa in zip(heads, C.holm(pvec)):
        res[m]["p_holm"] = float(pa)

    # ---- PAIRED between-head reliance diffs (image-clustered, Holm), on combined + each held-out operator ----
    names = list(heads)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    paired = {op: {} for op in ("combined", "step", "swap")}
    for op, key in (("combined", "reliance_combined"), ("step", "reliance_step"), ("swap", "reliance_swap")):
        ppv = []; keys = []
        for a, b in pairs:
            d, p = _paired(A[a][key], A[b][key])
            paired[op][f"{a}-vs-{b}"] = {"dReliance": d, "p": p}; ppv.append(p); keys.append(f"{a}-vs-{b}")
        for k, pa in zip(keys, C.holm(ppv)):
            paired[op][k]["p_holm"] = float(pa)

    # ---- TRANSFER verdict ----
    # cure effect = ordinalFT_reliance − cure_reliance (>0 ⇒ cure lowers reliance), and base − cure.
    def lo(op, pair):  # lower CI bound of the paired diff
        return paired[op][pair]["dReliance"][1]
    ft_vs_cure = "ordinalFT-vs-relightCure"; base_vs_cure = "base-vs-relightCure"
    transfer_vs_ft = bool(lo("combined", ft_vs_cure) > 0 and lo("step", ft_vs_cure) > 0 and lo("swap", ft_vs_cure) > 0)
    transfer_vs_base = bool(lo("combined", base_vs_cure) > 0 and lo("step", base_vs_cure) > 0 and lo("swap", base_vs_cure) > 0)
    cure_rel = res["relightCure"]["reliance_combined"]
    base_rel = res["base"]["reliance_combined"]
    ft_rel = res["ordinalFT"]["reliance_combined"]
    verdict = {
        "transfer_vs_ordinalFT": transfer_vs_ft,
        "transfer_vs_base": transfer_vs_base,
        "cure_reliance_below_zero": bool(cure_rel[2] < 0 or (cure_rel[1] <= 0 <= cure_rel[2])),
        "headline": ("TRANSFERS — the relight-cure's reliance reduction holds under the HELD-OUT operator "
                     "(cure < ordinal-FT on both held-out treatments, CI excludes 0) ⇒ genuine "
                     "brightness-invariance, not augmentation-memorization."
                     if transfer_vs_ft else
                     "DOES NOT TRANSFER — the cure's reliance reduction seen in-family is not reproduced "
                     "under the held-out operator (cure not significantly below ordinal-FT) ⇒ the reduction "
                     "is operator-specific / augmentation-memorization, not genuine invariance."),
        "cure_combined_reliance": cure_rel, "ordinalFT_combined_reliance": ft_rel, "base_combined_reliance": base_rel,
    }

    out = {
        "config": {"n_images": len(used), "amp": args.amp, "folds": "5-fold OOF (base=frozen head)",
                   "held_out_modes": ["uniform_bump→T_step", "scramble→T_swap+N_scram", "order_preserving→N_op"],
                   "training_operator": "relight mode='random' (finetune_dav2_consistency.py)",
                   "reliance_def": "reliance_op = swing_op − null_scram; combined = mean(step,swap)"},
        "models": res, "paired_reliance": paired, "verdict": verdict,
        "per_image": {m: {k: A[m][k].tolist() for k in KEYS} for m in heads}, "files": used,
    }
    outbase = os.path.join(P1, args.out)
    os.makedirs(os.path.dirname(outbase), exist_ok=True)
    json.dump(out, open(outbase + ".json", "w"), indent=1)
    _write_md(outbase + ".md", res, paired, verdict, args.amp, len(used))

    # ---- console summary ----
    print("\n=== held-out-operator reliance (reliance_op = swing_op − scramble-null; combined = mean of the two) ===")
    print(f"{'head':<13}{'clean-disc':>11}{'swing_step':>18}{'swing_swap':>18}{'nScram':>8}{'nOp':>8}"
          f"{'reliance_comb[CI]':>24}{'p_holm':>8}")
    for m in heads:
        r = res[m]
        rc = r["reliance_combined"]
        print(f"{m:<13}{r['clean_discordant']:>11.3f}"
              f"{r['swing_step'][0]:>+9.3f}[{r['swing_step'][1]:+.2f},{r['swing_step'][2]:+.2f}]"
              f"{r['swing_swap'][0]:>+9.3f}[{r['swing_swap'][1]:+.2f},{r['swing_swap'][2]:+.2f}]"
              f"{r['null_scram'][0]:>+8.3f}{r['null_op'][0]:>+8.3f}"
              f"{rc[0]:>+13.3f}[{rc[1]:+.2f},{rc[2]:+.2f}]{r['p_holm']:>8.3f}")
    print("\n--- paired between-head reliance Δ (combined; image-clustered, Holm) ---")
    for k, pr in paired["combined"].items():
        d = pr["dReliance"]
        print(f"  {k:<28} Δrel {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]  p_holm {pr['p_holm']:.3f}")
    print(f"\nVERDICT: {verdict['headline']}")
    print(f"wrote {args.out}.json / .md")


def _write_md(path, res, paired, verdict, amp, n):
    L = []
    L.append("# Reviewer #3.4 — does the relight-cure generalize to an INDEPENDENT (held-out) relight operator?\n")
    L.append(f"**{n} images · 5-fold out-of-fold · amp={amp} (matched to E1) · GPU forwards on relit images**\n")
    L.append("**Question.** The relight-consistency cure (`dav2cons`) was fit with brightness-invariance views "
             "from `relight.py` mode `random` (smooth Gaussian bumps, continuous random gains). The E1 reliance "
             "diagnostic uses `consistent`/`adversarial` — the SAME field family. Does the cure's reliance drop "
             "survive a HELD-OUT operator whose *gain law* it never trained on? Transfer ⇒ genuine "
             "brightness-invariance; failure ⇒ augmentation-memorization.\n")
    L.append("**Held-out family** (modes `uniform_bump`/`scramble`/`order_preserving`; same 5 points, sigma, amp — "
             "only the gain law changes): `T_step` = hard rank-threshold ±amp bumps (`uniform_bump`); "
             "`T_swap` = adversarial magnitudes permuted across points, sign follows depth order (`scramble`); "
             "`N_scram` = pure scramble null; `N_op` = global-gain null (`order_preserving`). "
             "reliance = swing − `N_scram`; combined = mean(step, swap).\n")
    L.append("## Held-out reliance per head\n")
    L.append("| head | clean-disc | swing_step [CI] | swing_swap [CI] | N_scram | N_op | reliance_combined [CI] | p_holm |")
    L.append("|---|---|---|---|---|---|---|---|")
    lab = {"base": "base (no FT)", "ordinalFT": "ordinal-FT (`dav2`)", "relightCure": "relight-cure (`dav2cons`)"}
    for m, r in res.items():
        rc = r["reliance_combined"]
        L.append(f"| {lab.get(m, m)} | {r['clean_discordant']:.3f} | "
                 f"{r['swing_step'][0]:+.3f} [{r['swing_step'][1]:+.2f},{r['swing_step'][2]:+.2f}] | "
                 f"{r['swing_swap'][0]:+.3f} [{r['swing_swap'][1]:+.2f},{r['swing_swap'][2]:+.2f}] | "
                 f"{r['null_scram'][0]:+.3f} | {r['null_op'][0]:+.3f} | "
                 f"**{rc[0]:+.3f}** [{rc[1]:+.2f},{rc[2]:+.2f}] | {r['p_holm']:.3f} |")
    L.append("\n_clean-disc = brightness-discordant pairwise accuracy on unrelit images (scoring-path guard vs "
             "RESULTS_finetune). N_scram / N_op ≈ 0 confirm the held-out apparatus emits no spurious swing._\n")
    L.append("## Paired between-head reliance Δ (image-clustered bootstrap, Holm)\n")
    for op in ("combined", "step", "swap"):
        L.append(f"**{op}**\n")
        L.append("| contrast | Δreliance [CI] | p_holm |")
        L.append("|---|---|---|")
        for k, pr in paired[op].items():
            d = pr["dReliance"]
            L.append(f"| {k} | {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] | {pr['p_holm']:.3f} |")
        L.append("")
    L.append("## Verdict\n")
    L.append(f"- **Transfer vs ordinal-FT:** {'YES' if verdict['transfer_vs_ordinalFT'] else 'NO'}  "
             "(cure < ordinal-FT on combined + both held-out treatments, CI excludes 0)")
    L.append(f"- **Transfer vs base:** {'YES' if verdict['transfer_vs_base'] else 'NO'}")
    L.append(f"- **Cure held-out reliance ≈/below 0:** {'YES' if verdict['cure_reliance_below_zero'] else 'NO'}\n")
    L.append(f"> {verdict['headline']}\n")
    open(path, "w").write("\n".join(L))


if __name__ == "__main__":
    main()
