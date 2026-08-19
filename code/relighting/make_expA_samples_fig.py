"""Compose the Exp A qualitative-samples figures from rendered panels (make_expA_sample_data.py).
  main  -> figures/F_expA_samples.png : one model (DAV2), one frame, RGB row + depth row across key interventions.
  supp  -> figures/F_expA_samples_supp.png : depth predictions for many models x interventions (and extra frames).
"""
import os, sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P1 = "/well/rittscher/users/fxh757/Code/Papers/P1_Ordinal_Endo_Depth_Benchmark__EndoLINA_MICCAI2026"
S = os.path.join(P1, "results/counterfactual/expA_samples")
FIGS = os.path.join(P1, "figures")
MLAB = {"dav2": "DAV2", "endoomni": "EndoOmni", "dac": "DAC", "metric3d": "Metric3D", "ppsnet": "PPSNet"}
FLAB = {"original": "original", "global": "global\nexposure", "spatial": "spatial\nfield",
        "physics": "near-field\nphysics", "decorr": "decorrelate", "invert": "invert"}
HARM = {"decorr", "invert"}       # red-labelled (coupling broken); others blue/grey


def load(p):
    im = cv2.imread(p)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im is not None else None


def _famcolor(f):
    return "#c53030" if f in HARM else ("#718096" if f == "physics" else "#2b6cb0")


def main_fig(model="dav2", fi=0, fams=("original", "global", "decorr", "invert")):
    ncol = len(fams) + 1                                    # + GT depth column
    fig, ax = plt.subplots(2, ncol, figsize=(2.0 * ncol, 4.15))
    for j, f in enumerate(fams):
        rgb = load(os.path.join(S, f"f{fi}_{f}_rgb.png")) if f != "original" else load(os.path.join(S, f"f{fi}_orig_rgb.png"))
        dep = load(os.path.join(S, model, f"f{fi}_{f}_dep.png"))
        for i, im in enumerate([rgb, dep]):
            ax[i, j].imshow(im if im is not None else np.ones((10, 10, 3)))
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            for sp in ax[i, j].spines.values():
                sp.set_edgecolor(_famcolor(f)); sp.set_linewidth(1.8)
        ax[0, j].set_title(FLAB[f], fontsize=9, color=_famcolor(f))
    # GT depth reference column
    gt = load(os.path.join(S, f"f{fi}_gt.png"))
    ax[1, ncol - 1].imshow(gt if gt is not None else np.ones((10, 10, 3)))
    ax[1, ncol - 1].set_title("GT depth", fontsize=9, color="#2d3748")
    ax[0, ncol - 1].axis("off")
    ax[1, ncol - 1].set_xticks([]); ax[1, ncol - 1].set_yticks([])
    ax[0, 0].set_ylabel("relit RGB", fontsize=9); ax[1, 0].set_ylabel(f"{MLAB.get(model, model)} depth", fontsize=9)
    for a in [ax[0, 0], ax[1, 0]]:
        a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(w_pad=0.3, h_pad=0.4)
    out = os.path.join(FIGS, "F_expA_samples.png")
    plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
    print("wrote", out)


def supp_fig(models=("dav2", "endoomni", "dac", "metric3d", "ppsnet"), fi=0,
             fams=("original", "global", "spatial", "physics", "decorr", "invert")):
    nrow = len(models) + 1                                  # + RGB row on top
    ncol = len(fams)
    fig, ax = plt.subplots(nrow, ncol, figsize=(1.7 * ncol, 1.7 * nrow))
    for j, f in enumerate(fams):
        rgb = load(os.path.join(S, f"f{fi}_{f}_rgb.png")) if f != "original" else load(os.path.join(S, f"f{fi}_orig_rgb.png"))
        ax[0, j].imshow(rgb if rgb is not None else np.ones((10, 10, 3)))
        ax[0, j].set_title(FLAB[f], fontsize=8.5, color=_famcolor(f))
        for i, m in enumerate(models):
            dep = load(os.path.join(S, m, f"f{fi}_{f}_dep.png"))
            ax[i + 1, j].imshow(dep if dep is not None else np.ones((10, 10, 3)))
        for i in range(nrow):
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            for sp in ax[i, j].spines.values():
                sp.set_edgecolor(_famcolor(f)); sp.set_linewidth(1.4)
    ax[0, 0].set_ylabel("relit RGB", fontsize=8.5)
    for i, m in enumerate(models):
        ax[i + 1, 0].set_ylabel(MLAB.get(m, m), fontsize=8.5)
    plt.tight_layout(w_pad=0.2, h_pad=0.2)
    out = os.path.join(FIGS, "F_expA_samples_supp.png")
    plt.savefig(out, dpi=190, bbox_inches="tight"); plt.close()
    print("wrote", out)


if __name__ == "__main__":
    fi = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main_fig(fi=fi)
    supp_fig(fi=fi)
