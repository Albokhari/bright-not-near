"""§16 spine figure — the one unifying result: causal brightness reliance (E1 swing) across models/cures.

Bars = E1 reliance (swing − null) with image-clustered 95% CI, ordered to tell the story:
  base DAV2  →  §13-cured (accuracy↑ but reliance not lowered)  →  N4-consistency-cured (reliance↓, the cure)
  →  EndoOmni (shortcut-robust foundation reference).
Reads results/counterfactual/E1_reliance.json (+ E1_n4.json if present). Shared §7 color tokens.
Run (CPU): python3 make_spine_figure.py
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "validation"))
import p1_common as C

RES = os.path.join(C.P1, "results/counterfactual")
FIGDIR = os.path.join(C.P1, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# display order, label, color (§7 tokens: DAV2 orange foil, cures teal ramp, EndoOmni violet)
SPEC = [
    ("DAV2-base", "DAV2\n(base)", "#E8873A"),
    ("DAV2-s13", "+ §13\nordinal FT", "#9ecae1"),
    ("DAV2-n4", "+ N4\nconsistency", "#2b8cbe"),
    ("EndoOmni", "EndoOmni\n(foundation)", "#7B5EA7"),
]


def load_reliance():
    j = json.load(open(os.path.join(RES, "E1_reliance_final.json")))
    return j["models"], j.get("paired_swing", {})


def main():
    r, paired = load_reliance()
    rows = [(lab, col, r[key]["reliance_swing"]) for key, lab, col in SPEC if key in r]
    labs = [x[0] for x in rows]; cols = [x[1] for x in rows]
    vals = np.array([x[2][0] for x in rows]); lo = np.array([x[2][1] for x in rows]); hi = np.array([x[2][2] for x in rows])

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    xpos = np.arange(len(rows))
    ax.bar(xpos, vals, color=cols, width=0.62, edgecolor="black", linewidth=0.8, zorder=3)
    ax.errorbar(xpos, vals, yerr=[vals - lo, hi - vals], fmt="none", ecolor="black",
                elinewidth=1.2, capsize=4, zorder=4)
    for x, v, h in zip(xpos, vals, hi):
        ax.text(x, h + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    # paired-significance bracket: §13 vs N4 (the consistency-cure effect)
    keys = [s[0] for s in SPEC if s[0] in r]
    if "DAV2-s13" in keys and "DAV2-n4" in keys:
        i, jx = keys.index("DAV2-s13"), keys.index("DAV2-n4")
        pk = "DAV2-s13-vs-DAV2-n4"
        ph = paired.get(pk, {}).get("p_holm", 1.0)
        yb = max(hi[i], hi[jx]) + 0.012
        ax.plot([i, i, jx, jx], [hi[i] + 0.004, yb, yb, hi[jx] + 0.004], color="black", lw=0.9)
        star = "*" if ph < 0.05 else "n.s."
        ax.text((i + jx) / 2, yb + 0.001, f"N4<§13 {star}", ha="center", va="bottom", fontsize=7.5)
    # EndoOmni reference band (the shortcut-robust target)
    if "EndoOmni" in r:
        eo = r["EndoOmni"]["reliance_swing"][0]
        ax.axhline(eo, ls="--", color="#7B5EA7", lw=1.1, zorder=2)
        ax.text(len(rows) - 0.5, eo - 0.004, "foundation-model floor (sig. lowest)", ha="right", va="top",
                fontsize=7.5, color="#7B5EA7")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xpos); ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylabel("causal brightness reliance\n(relight swing = acc[consistent] − acc[adversarial])", fontsize=9)
    ax.set_title("Counterfactual-relight reliance: diagnose the shortcut, test the cures", fontsize=9.5)
    ax.set_ylim(0, max(hi) * 1.32)
    ax.grid(axis="y", ls=":", alpha=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"F_spine_reliance.{ext}"), dpi=200, bbox_inches="tight")
    print("wrote figures/F_spine_reliance.{pdf,png}")
    print("reliance:", {lab: round(v, 3) for lab, v in zip(labs, vals)})


if __name__ == "__main__":
    main()
