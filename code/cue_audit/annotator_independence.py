"""Review-response earn (#3, reviewer Q10): annotator-independence checks.

(1) LEAVE-ONE-ANNOTATOR-OUT (LOAO): drop each of the 4 raters, recompute mean pairwise Kendall tau + Kendall's W
    among the remaining raters (on PASS-307 and ALL-400). If agreement is stable to dropping any single rater, the
    consensus is not carried by one rater.
(2) CONCORDANT-vs-DISCORDANT inter-annotator agreement: for each 5-point image, split the 10 point-pairs by whether
    brightness AGREES (concordant: the GT-nearer point is brighter) or DISAGREES (adversarial: GT-nearer is darker)
    with depth, and measure INTER-ANNOTATOR agreement (fraction of rater-pairs ordering that pair identically) on
    each half. The reviewer's Q10 = "does human error correlate with brightness?": if annotators agree ~equally on
    both halves, the human signal is geometric even where brightness misleads; if they agree much LESS on the
    adversarial half, disagreement concentrates where brightness conflicts (an honest bound to report).

Reuses compute_iaa's raw-order loading + p1_common. Output: results/validation/annotator_indep/{annotator_indep.json, .md}
Run (CPU short): python3 annotator_independence.py
"""
import os, json, itertools
import numpy as np
import cv2
from scipy.stats import kendalltau
import p1_common as C

CG = os.path.join(C.P1, "results/consensus_GT")
IMG = "/well/rittscher/users/fxh757/Code/Kvasir_Ranking/Merged_PASS/images"
OUT = os.path.join(C.P1, "results/validation/annotator_indep"); os.makedirs(OUT, exist_ok=True)
BLOCKS = {"Annotator1": "annotator1", "Annotator2": "annotator2", "Annotator4": "annotator4", "Annotator3": "annotator3"}
RATERS = ["annotator1", "annotator2", "annotator4", "annotator3"]


def kendalls_w(rankings):
    R = np.asarray(rankings, float); m, n = R.shape
    col = R.sum(0); S = ((col - col.mean()) ** 2).sum()
    denom = m ** 2 * (n ** 3 - n) / 12.0
    return S / denom if denom > 0 else np.nan


def bright_at(fn, pts):
    bgr = cv2.imread(os.path.join(IMG, fn))
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(float)
    g = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    out = []
    for (x, y) in pts:
        out.append(g[min(int(y), g.shape[0] - 1), min(int(x), g.shape[1] - 1)])
    return np.array(out)


def load_all():
    """Merge decisions across blocks -> {fn: [rankvec,...]} (annotator labels are inconsistent across blocks, so we
    keep the SET of 4 orders per image, deduped), plus merged GT points/ranks."""
    merged = {im["filename"]: im for im in json.load(open(os.path.join(CG, "merged_pass_consensus_gt.json")))["images"]}
    pass_files = set(merged)
    byimg = {}
    for block, key in BLOCKS.items():
        data = json.load(open(os.path.join(CG, f"{key}_consensus_summary.json")))
        for dec in data["decisions"]:
            fn = dec["image_name"]
            if fn in byimg:                                    # each image is rated in ONE block; take first
                continue
            orders = [list(o) for o in dec.get("annotator_orders", {}).values() if len(o) == 5]
            if orders:                                         # keep ALL 4 raters (do NOT dedup identical = agreement!)
                byimg[fn] = orders
    return byimg, pass_files, merged


def _mtau(orders):
    return float(np.nanmean([kendalltau(a, b).correlation for a, b in itertools.combinations(orders, 2)]))


def loao(byimg, subset):
    """Leave-one-out ROBUSTNESS (annotator labels inconsistent across blocks -> drop by position, not identity):
    per image, drop each of the 4 orders and recompute agreement. Report all-rater vs leave-one-out mean/worst."""
    allt, allw, lom, low = [], [], [], []
    for fn in subset:
        orders = byimg.get(fn, [])
        if len(orders) < 3:
            continue
        allt.append(_mtau(orders)); allw.append(kendalls_w(orders))
        drops = [_mtau([o for x, o in enumerate(orders) if x != k]) for k in range(len(orders))]
        lom.append(float(np.mean(drops))); low.append(float(np.min(drops)))
    return {"n": len(allt), "all_tau": round(float(np.nanmean(allt)), 4), "all_W": round(float(np.nanmean(allw)), 4),
            "loo_tau_mean": round(float(np.nanmean(lom)), 4), "loo_tau_worst": round(float(np.nanmean(low)), 4)}


def conc_disc_agreement(byimg, merged, subset):
    """Inter-annotator agreement on brightness-concordant vs brightness-adversarial point-pairs."""
    conc_agree, disc_agree = [], []
    for fn in subset:
        orders = byimg.get(fn, [])
        if len(orders) < 2 or fn not in merged:
            continue
        pts = sorted(merged[fn]["points"], key=lambda p: p["point_id"])
        xy = [(p["x"], p["y"]) for p in pts]
        gtrank = np.array([p["rank"] for p in pts], float)         # 1=nearest
        b = bright_at(fn, xy)
        if b is None:
            continue
        for i, j in itertools.combinations(range(5), 2):
            # GT: which is nearer? (smaller rank). concordant if the nearer point is brighter.
            near = i if gtrank[i] < gtrank[j] else j
            far = j if near == i else i
            concordant = b[near] >= b[far]
            # inter-annotator agreement on ordering of (i,j): fraction of rater-pairs agreeing
            signs = [np.sign(o[i] - o[j]) for o in orders]
            agr = np.mean([signs[a] == signs[c] for a, c in itertools.combinations(range(len(signs)), 2)])
            (conc_agree if concordant else disc_agree).append(agr)
    return (round(float(np.mean(conc_agree)), 4), len(conc_agree),
            round(float(np.mean(disc_agree)), 4), len(disc_agree))


def main():
    byimg, pass_files, merged = load_all()
    all_files = set(byimg)
    out = {"loao_pass": loao(byimg, pass_files), "loao_all": loao(byimg, all_files)}
    ca, cn, da, dn = conc_disc_agreement(byimg, merged, pass_files)
    out["concordant_vs_discordant"] = {"concordant_agree": ca, "n_concordant_pairs": cn,
                                       "discordant_agree": da, "n_discordant_pairs": dn,
                                       "gap": round(ca - da, 4)}
    json.dump(out, open(os.path.join(OUT, "annotator_indep.json"), "w"), indent=1)

    lp, la = out["loao_pass"], out["loao_all"]
    lines = ["# Annotator-independence (review-response #3 / Q10)\n",
             "## Leave-one-out robustness — mean pairwise Kendall tau / Kendall's W",
             "| subset | n | all-rater tau | W | leave-one-out tau (mean) | leave-one-out tau (worst drop) |",
             "|---|---|---|---|---|---|",
             f"| PASS-307 | {lp['n']} | {lp['all_tau']} | {lp['all_W']} | {lp['loo_tau_mean']} | {lp['loo_tau_worst']} |",
             f"| ALL-400 | {la['n']} | {la['all_tau']} | {la['all_W']} | {la['loo_tau_mean']} | {la['loo_tau_worst']} |",
             "(loo tau close to all-rater tau ⇒ agreement is not carried by any single annotator)",
             "", "## Inter-annotator agreement: brightness-concordant vs adversarial point-pairs (PASS)",
              f"- concordant pairs (GT-nearer is brighter): agreement **{ca}** (n={cn})",
              f"- adversarial pairs (GT-nearer is darker):  agreement **{da}** (n={dn})",
              f"- gap (conc - disc) = **{round(ca-da,4)}**",
              "",
              "Interpretation: a SMALL gap ⇒ annotators agree even where brightness misleads (human signal is",
              "geometric, not brightness-driven); a LARGE gap ⇒ disagreement concentrates on adversarial pairs",
              "(honest bound on how much of the label signal is brightness-confident geometry)."]
    open(os.path.join(OUT, "annotator_indep.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT}/annotator_indep.{{json,md}}")


if __name__ == "__main__":
    main()
