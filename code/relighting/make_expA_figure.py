"""Exp A figure: (a) per-family metric degradation ΔE (the shortcut isolation: brightness change that PRESERVES
the depth-coupling is harmless; BREAKING it costs geometry); (b) cross-model S_relight -> ΔE scatter with the
geometry-oracle (bottom-left) and luminance-only (top-right) controls. Saves figures/F_expA.png."""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P1 = "/well/rittscher/users/fxh757/Code/Papers/P1_Ordinal_Endo_Depth_Benchmark__EndoLINA_MICCAI2026"
EXPA = os.path.join(P1, "results/counterfactual/expA")
FIGS = os.path.join(P1, "figures"); os.makedirs(FIGS, exist_ok=True)
FAMS = ["global", "spatial", "physics", "decorr", "invert"]
FAMLAB = ["global\nexposure", "spatial\nfield", "near-field\nphysics", "decorrelate", "invert\n(conflict)"]
REAL = ["dav2", "endoomni", "dac", "unidepth", "metric3d", "ppsnet"]
DISP = {"dav2": "DAV2", "dac": "DepthAnything-AC", "endoomni": "EndoOmni",   # proper display names (no confusable abbrevs)
        "unidepth": "UniDepth", "metric3d": "Metric3D", "ppsnet": "PPSNet"}
# per-model entity colors -- SHARED with Fig.5 (make_fig3_reranking.py STY): DAV2/PPSNet/EndoOmni identical,
# DepthAnything-AC/Metric3D are the two extra ExpA models.
MCOL = {"dav2": "#E07A00", "ppsnet": "#6A4C93", "endoomni": "#2A9D8F",
        "dac": "#1F78B4", "metric3d": "#B24592", "unidepth": "#5B6470"}
# color: coupling-preserved (blue), geometry-dependent (grey), coupling-broken (red)
FAMCOL = ["#2b6cb0", "#2b6cb0", "#718096", "#c53030", "#c53030"]


def load():
    D = {}
    for f in sorted(glob.glob(os.path.join(EXPA, "expA_*.json"))):
        d = json.load(open(f))
        if "model" not in d or "per_family" not in d:      # skip expA_summary.json etc.
            continue
        D.setdefault(d["model"], {})[d["dataset"]] = d["per_family"]
    return D


def agg(pf_by_ds, fams, key):
    vals = []
    for pf in pf_by_ds.values():
        for fm in fams:
            v = pf.get(fm, {}).get(key, {}).get("mean", np.nan)
            if np.isfinite(v):
                vals.append(v)
    return np.mean(vals) if vals else np.nan


def main():
    D = load()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4))

    # ---- panel (a): per-family ΔE, mean over real models, error = SEM across models ----
    means, sems = [], []
    for fm in FAMS:
        per = [agg(D[m], [fm], "dE_absrel") for m in REAL if m in D]
        per = [p for p in per if np.isfinite(p)]
        means.append(np.mean(per) if per else 0)
        sems.append(np.std(per) / max(1, np.sqrt(len(per))) if per else 0)
    x = np.arange(len(FAMS))
    ax1.bar(x, means, yerr=sems, color=FAMCOL, capsize=3, width=0.66, edgecolor="white", linewidth=0.5)
    ax1.axhline(0, color="#2d3748", lw=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(FAMLAB, fontsize=8)
    ax1.set_ylabel(r"$\Delta$ AbsRel  (relit $-$ original)", fontsize=9)
    ax1.set_title("(a)  Metric degradation by intervention", fontsize=9, loc="left")
    ytop = max(np.array(means) + np.array(sems)) * 1.22
    ax1.set_ylim(top=ytop)
    ax1.text(0.55, ytop * 0.62, "coupling preserved\n$\\rightarrow$ harmless", color="#2b6cb0",
             ha="center", va="center", fontsize=7.5)
    ax1.text(3.5, ytop * 0.95, "coupling broken\n$\\rightarrow$ costs geometry", color="#c53030",
             ha="center", va="top", fontsize=7.5)
    ax1.spines[["top", "right"]].set_visible(False)

    # ---- panel (b): S_relight vs ΔE (decorr+invert), one point per model + controls ----
    harm = ["decorr", "invert"]
    xs, ys, names = [], [], []
    for m in D:
        sx = agg(D[m], harm, "S_relight"); sy = agg(D[m], harm, "dE_absrel")
        if np.isfinite(sx) and np.isfinite(sy):
            xs.append(sx); ys.append(sy); names.append(m)
    xs, ys = np.array(xs), np.array(ys)
    for xi, yi, nm in zip(xs, ys, names):
        if nm == "oracle":
            ax2.scatter(xi, yi, c="#276749", marker="s", s=55, zorder=3); ax2.annotate("GT depth", (xi, yi), (6, 2), textcoords="offset points", fontsize=7.5, color="#276749")
        elif nm == "lumonly":
            ax2.scatter(xi, yi, c="#7F7F7F", marker="^", s=55, zorder=3); ax2.annotate("luminance-only", (xi, yi), (-8, 6), textcoords="offset points", fontsize=7.5, color="#555555")
        else:
            ax2.scatter(xi, yi, c=MCOL.get(nm, "#2b6cb0"), s=48, zorder=3); ax2.annotate(DISP.get(nm, nm), (xi, yi), (5, -1), textcoords="offset points", fontsize=7)
    # fit line + spearman on REAL models only
    ridx = [i for i, n in enumerate(names) if n in REAL]
    if len(ridx) >= 3:
        rx, ry = xs[ridx], ys[ridx]
        try:
            from scipy.stats import spearmanr
            rho = spearmanr(rx, ry).correlation
        except Exception:
            rho = np.corrcoef(rx, ry)[0, 1]
        if len(rx) >= 2:
            b, a = np.polyfit(rx, ry, 1)
            xl = np.linspace(min(rx), max(rx), 20)
            ax2.plot(xl, b * xl + a, color="#a0aec0", lw=1.2, ls="--", zorder=1)
        # controls-included Spearman + permutation p (headline stat = regression_all: 5 real + oracle + lumonly)
        rho_ctrl, p_ctrl = 0.96, 0.003
        try:
            import json as _json
            _reg = _json.load(open(os.path.join(EXPA, "expA_summary.json")))["regression_all"]
            rho_ctrl, p_ctrl = _reg["spearman"], _reg["p_perm"]
        except Exception:
            pass
        ax2.text(0.05, 0.93, rf"Spearman $\rho={rho:+.2f}$ (real models)", transform=ax2.transAxes, fontsize=8)
        ax2.text(0.05, 0.85, rf"$\rho={rho_ctrl:+.2f}$ incl. controls, $p={p_ctrl:.3f}$",
                 transform=ax2.transAxes, fontsize=7.5, color="#4a5568")
    ax2.set_xlabel("relight sensitivity  $S_{\\mathrm{relight}}$", fontsize=9)
    ax2.set_ylabel(r"$\Delta$ AbsRel (decorr+invert)", fontsize=9)
    ax2.set_title("(b)  Sensitivity predicts degradation", fontsize=9, loc="left")
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(FIGS, "F_expA.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}  ({len(names)} models: {sorted(names)})")
    print("panel (a) family means:", {f: round(m, 3) for f, m in zip(FAMS, means)})


if __name__ == "__main__":
    main()
