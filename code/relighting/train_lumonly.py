"""Train the Exp A `lumonly` positive control: a tiny grayscale-input CNN (TinyLumNet, defined in
expA_controls.py) that predicts a single-channel relative-depth map from BRIGHTNESS ALONE.

We train on C3VD-train (the *distorted* Depth_split_2/train, matching the distorted Depth_split_2/test
that expA_run uses) with a scale-and-shift-invariant loss (per-image least-squares affine align of the
prediction to the GT depth, then residual MSE normalised by GT variance) -- the same ss alignment the
eval uses. Because C3VD is lit bright~=near, a grayscale-only net *learns to follow brightness*, which is
its entire purpose: under the `decorr` relight family it should then break (high dE, high S_relight).

The net is tiny (~0.2M params) so the whole encoder+head is trained end-to-end for a few epochs.

Run (GPU) -- see the sbatch block at the bottom of this file, or directly:
  module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1 SciPy-bundle
  python3 train_lumonly.py --epochs 15 --bs 8

Writes: results/finetune/lumonly/lumonly.pt   (state_dict + input_size + preprocessing constants)
"""
import os, sys, json, argparse, time
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import expA_controls as EC   # TinyLumNet, gray_from_bgr, INPUT_SIZE, GRAY_MEAN/STD, LUMONLY_CKPT

P1 = EC.P1
C3VD_TRAIN = "/well/rittscher/users/fxh757/Datasets/PUBLIC_ENDO/Depth_split_2/train"
SPLIT_JSON = os.path.join(P1, "results/finetune/c3vd_train_ord_split.json")
EPS = 1e-6


# --------------------------------------------------------------------------------------
# Data: grayscale image -> GT depth (larger=farther), both resized to INPUT_SIZE.
# --------------------------------------------------------------------------------------
class C3VDGrayDepth(Dataset):
    def __init__(self, files, size=EC.INPUT_SIZE, root=C3VD_TRAIN):
        self.files = files
        self.size = size
        self.img_dir = os.path.join(root, "images")
        self.dep_dir = os.path.join(root, "depths")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fn = self.files[i]
        bgr = cv2.imread(os.path.join(self.img_dir, fn), cv2.IMREAD_COLOR)
        dep = cv2.imread(os.path.join(self.dep_dir, fn), cv2.IMREAD_UNCHANGED)
        if bgr is None or dep is None:
            # degenerate placeholder (fully masked-out) so a bad file cannot crash the loader
            z = np.zeros((self.size, self.size), np.float32)
            return (torch.zeros(1, self.size, self.size), torch.from_numpy(z), torch.zeros_like(torch.from_numpy(z)))
        g = EC.gray_from_bgr(bgr)                                  # HxW [0,1]
        g = cv2.resize(g, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        g = (g - EC.GRAY_MEAN) / EC.GRAY_STD
        d = dep.astype(np.float32)                                # larger=farther, 0=invalid
        d = cv2.resize(d, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        mask = (d > 0).astype(np.float32)
        return (torch.from_numpy(g)[None].float(),
                torch.from_numpy(d).float(),
                torch.from_numpy(mask).float())


def list_split(use_all, limit):
    """Return (train_files, val_files). Default: the documented ord split; --use_all uses every train frame."""
    img_dir = os.path.join(C3VD_TRAIN, "images")
    if (not use_all) and os.path.exists(SPLIT_JSON):
        sp = json.load(open(SPLIT_JSON))
        tr = [f for f in sp["train"] if os.path.exists(os.path.join(img_dir, f))]
        va = [f for f in sp.get("val", []) if os.path.exists(os.path.join(img_dir, f))]
    else:
        allf = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".png"))
        rng = np.random.default_rng(0)
        rng.shuffle(allf)
        nval = max(20, int(0.05 * len(allf)))
        va, tr = allf[:nval], allf[nval:]
    if limit:
        tr, va = tr[:limit], va[: max(1, limit // 5)]
    return tr, va


# --------------------------------------------------------------------------------------
# Scale-and-shift-invariant loss: per-image least-squares affine align pred->gt, residual MSE / var(gt).
# Fully differentiable in `pred`; `a` may be negative (disparity-vs-depth sign is free), matching ss_fit.
# --------------------------------------------------------------------------------------
def ss_invariant_loss(pred, gt, mask):
    """Scale-and-shift-invariant (sign-free) residual, per-image.

    We standardise the GT to zero-mean/unit-std over the valid mask, then least-squares affine-align the
    prediction to it (slope `a` free incl. negative -> disparity-vs-depth sign is absorbed, matching
    expA_metrics.ss_fit) and return the residual MSE. Standardising the GT bounds the residual in ~[0,1]
    (it equals 1 - r^2 at the LS optimum), which is numerically stable and interpretable, unlike dividing a
    raw-unit residual by a raw-unit variance. pred,gt,mask: (B,1,H,W) -> scalar batch mean."""
    B = pred.shape[0]
    p = pred.reshape(B, -1)
    g = gt.reshape(B, -1)
    m = (mask.reshape(B, -1) > 0.5).float()
    n = m.sum(1).clamp(min=1.0)
    # standardise GT per image over the valid mask -> unit scale (keeps invariance, bounds the residual)
    mg = (g * m).sum(1) / n
    var_g = ((g - mg[:, None]) ** 2 * m).sum(1) / n
    gn = (g - mg[:, None]) / (var_g.sqrt()[:, None] + EPS)
    # least-squares affine fit p -> gn (sign-free), then residual on the standardised target
    mp = (p * m).sum(1) / n
    mgn = (gn * m).sum(1) / n
    pc = (p - mp[:, None]) * m
    gc = (gn - mgn[:, None]) * m
    var_p = (pc * pc).sum(1) / n
    cov = (pc * gc).sum(1) / n                                   # mean, consistent with var_p (both /n)
    a = cov / (var_p + 1e-6 * var_p.mean() + EPS)               # relative eps -> no blow-up when p ~ const
    b = mgn - a * mp
    aligned = a[:, None] * p + b[:, None]
    resid = ((aligned - gn) ** 2 * m).sum(1) / n                 # in ~[0,1]; = 1 - r^2 at LS optimum
    valid = (m.sum(1) > 50).float()                              # skip near-empty frames
    return (resid * valid).sum() / valid.sum().clamp(min=1.0)


def evaluate(net, loader, device):
    net.eval()
    tot, nb = 0.0, 0
    with torch.no_grad():
        for g, d, m in loader:
            g, d, m = g.to(device), d.to(device)[:, None], m.to(device)[:, None]
            loss = ss_invariant_loss(net(g), d, m)
            tot += float(loss); nb += 1
    return tot / max(nb, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--size", type=int, default=EC.INPUT_SIZE)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--use_all", action="store_true", help="train on ALL Depth_split_2/train frames (default: ord split)")
    ap.add_argument("--limit", type=int, default=0, help="cap #train frames (smoke)")
    ap.add_argument("--out", default=EC.LUMONLY_CKPT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tr, va = list_split(args.use_all, args.limit)
    print(f"[train_lumonly] train={len(tr)} val={len(va)} size={args.size} dev={args.device} "
          f"root={C3VD_TRAIN}", flush=True)
    assert len(tr) > 0, f"no training frames found under {C3VD_TRAIN}/images"

    dtr = DataLoader(C3VDGrayDepth(tr, args.size), batch_size=args.bs, shuffle=True,
                     num_workers=args.workers, drop_last=True, pin_memory=(args.device == "cuda"))
    dva = DataLoader(C3VDGrayDepth(va, args.size), batch_size=args.bs, shuffle=False,
                     num_workers=args.workers) if va else None

    net = EC.TinyLumNet().to(args.device)
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"[train_lumonly] TinyLumNet params={nparam/1e6:.3f}M | ss-invariant loss | {args.epochs} epochs", flush=True)

    best = float("inf")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ep in range(args.epochs):
        net.train(); t0 = time.time(); run, nb = 0.0, 0
        for g, d, m in dtr:
            g, d, m = g.to(args.device), d.to(args.device)[:, None], m.to(args.device)[:, None]
            opt.zero_grad()
            loss = ss_invariant_loss(net(g), d, m)
            loss.backward()
            opt.step()
            run += float(loss); nb += 1
        sched.step()
        trl = run / max(nb, 1)
        val = evaluate(net, dva, args.device) if dva else trl
        tag = ""
        if val <= best:
            best = val
            torch.save({"state_dict": net.state_dict(), "input_size": args.size,
                        "gray_mean": EC.GRAY_MEAN, "gray_std": EC.GRAY_STD,
                        "arch": "TinyLumNet", "loss": "ss_invariant", "val": val,
                        "note": "lumonly positive control (brightness->relative depth) for Exp A"},
                       args.out)
            tag = "  *saved(best)"
        print(f"  ep {ep+1:02d}/{args.epochs}  train {trl:.4f}  val {val:.4f}  "
              f"lr {sched.get_last_lr()[0]:.2e}  {time.time()-t0:.1f}s{tag}", flush=True)

    print(f"[train_lumonly] done. best val {best:.4f} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------------------------------
# sbatch (GPU) — save as job_lumonly.sh or submit inline:
#   #!/bin/bash
#   #SBATCH -p short --gres=gpu:1 -c 4 --mem 24G -t 01:00:00 -J lumonly
#   #SBATCH -o lumonly_%j.out -e lumonly_%j.err
#   module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1 SciPy-bundle
#   cd /well/rittscher/users/fxh757/Code/Papers/P1_Ordinal_Endo_Depth_Benchmark__EndoLINA_MICCAI2026/code/counterfactual
#   python3 train_lumonly.py --epochs 15 --bs 8 --use_all
# ---------------------------------------------------------------------------------------------------
