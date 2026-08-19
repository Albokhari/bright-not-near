"""DAV2 head-only fine-tune with DIW ordinal ranking loss (pilot).

- Loads DAV2 ViT-L, FREEZES the DINOv2 encoder (features computed under no_grad),
  trains only the DPT head.
- Per training image: forward -> depth map; sample 5 annotated points (differentiable
  grid_sample); nearness score = depth (DAV2: larger = nearer); DIW ranking loss with
  optional brightness-adversarial pair up-weighting.
- Early-stop on val ordinal ExactMatch. Saves best head.
- After training: scores BASE vs FINETUNED on the held-out TEST fold and dumps a
  per-image/per-point CSV (for image-clustered stats) + a summary.

Run on a GPU node (see finetune_pilot.sbatch). Outputs: results/finetune/<run>/.
"""
import os, sys, json, argparse, csv
import numpy as np
import cv2
import torch
import torch.nn.functional as F

DA = "/well/rittscher/users/fxh757/Code/depth_anything_v2"
sys.path.insert(0, DA)
sys.path.insert(0, os.path.dirname(__file__))
from depth_anything_v2.dpt import DepthAnythingV2  # noqa
from ranking_loss import diw_ranking_loss, pair_list  # noqa

P1 = "/well/rittscher/users/fxh757/Code/Papers/P1_Ordinal_Endo_Depth_Benchmark__EndoLINA_MICCAI2026"
GT = json.load(open(os.path.join(P1, "results/consensus_GT/merged_pass_consensus_gt.json")))
GTD = {im["filename"]: im for im in GT["images"]}
FOLDS = json.load(open(os.environ.get("P1_FOLDS", os.path.join(P1, "results/finetune/folds.json"))))  # $P1_FOLDS overrides (e.g. cluster-disjoint folds)
IMG_ROOT = "/well/rittscher/users/fxh757/Code/Kvasir_Ranking/Merged_PASS/images"
CKPT = os.path.join(DA, "checkpoints/depth_anything_v2_vitl.pth")
CFG = {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]}
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(head_state=None):
    m = DepthAnythingV2(**CFG)
    m.load_state_dict(torch.load(CKPT, map_location="cpu"))
    if head_state is not None:
        m.depth_head.load_state_dict(head_state)
    m = m.to(DEV).eval()
    for p in m.pretrained.parameters():
        p.requires_grad_(False)
    return m


def _img2tensor(bgr, input_size):
    # replicate DAV2 image2tensor without needing a model instance
    from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
    from torchvision.transforms import Compose
    tr = Compose([Resize(width=input_size, height=input_size, resize_target=False,
                         keep_aspect_ratio=True, ensure_multiple_of=14,
                         resize_method="lower_bound", image_interpolation_method=cv2.INTER_CUBIC),
                  NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                  PrepareForNet()])
    h, w = bgr.shape[:2]
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.0
    img = tr({"image": img})["image"]
    return torch.from_numpy(img).unsqueeze(0).to(DEV), (h, w)


def forward_depth(model, x, train_head):
    patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
    with torch.no_grad():
        feats = model.pretrained.get_intermediate_layers(
            x, model.intermediate_layer_idx[model.encoder], return_class_token=True)
    feats = [(a.detach(), b.detach()) for (a, b) in feats]
    ctx = torch.enable_grad() if train_head else torch.no_grad()
    with ctx:
        depth = model.depth_head(feats, patch_h, patch_w).squeeze(1)  # [1,H',W']
    return depth[0]  # [H',W']


def sample_points(depth, pts_xy, orig_hw, input_size):
    """Differentiable bilinear sample of nearness scores at the 5 points.
    depth: [H',W'] tensor; pts_xy: (5,2) original-res coords; returns (5,) scores."""
    Hh, Ww = depth.shape
    oh, ow = orig_hw
    gx = (pts_xy[:, 0] / ow) * 2 - 1   # normalize to [-1,1]
    gy = (pts_xy[:, 1] / oh) * 2 - 1
    grid = torch.tensor(np.stack([gx, gy], 1), dtype=depth.dtype, device=depth.device).view(1, 1, 5, 2)
    s = F.grid_sample(depth.view(1, 1, Hh, Ww), grid, mode="bilinear", align_corners=True)
    return s.view(5)


CROP = False   # set True via --crop: center-crop(518) input matching the C3VD eval preprocessing


def load_item(fn):
    if CROP:
        import crop_kvasir as CK
        bgr, xy, ranks, bright, _ = CK.cropped_load(IMG_ROOT, fn, GTD[fn])
        return bgr, xy, ranks, bright, (CK.CROP, CK.CROP)
    bgr = cv2.imread(os.path.join(IMG_ROOT, fn))
    im = GTD[fn]
    pts = sorted(im["points"], key=lambda p: p["point_id"])
    xy = np.array([[p["x"], p["y"]] for p in pts], float)
    ranks = np.array([p["rank"] for p in pts], float)
    # brightness at each point (grayscale of original RGB)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(float)
    g = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    bx = np.clip(xy[:, 0].astype(int), 0, g.shape[1] - 1)
    by = np.clip(xy[:, 1].astype(int), 0, g.shape[0] - 1)
    bright = g[by, bx]
    return bgr, xy, ranks, bright, (bgr.shape[0], bgr.shape[1])


def ordinal_metrics(scores, ranks, bright):
    """scores: np (5,) nearness. returns exact, pairwise, discordant_acc, n_disc."""
    s = np.asarray(scores); r = np.asarray(ranks)
    pred_order = np.argsort(-s)              # nearest first
    gt_order = np.argsort(r)
    exact = float(np.array_equal(pred_order, gt_order))
    pc = pn = dc = dn = 0
    for (i, j) in pair_list(r):              # i nearer than j
        ok = s[i] > s[j]
        pc += ok; pn += 1
        if bright[i] < bright[j]:            # brightness-adversarial pair
            dc += ok; dn += 1
    return exact, pc / pn, (dc / dn if dn else np.nan), dn


@torch.no_grad()
def score_set(model, files, input_size):
    rows = []
    for fn in files:
        bgr, xy, ranks, bright, ohw = load_item(fn)
        x, _ = _img2tensor(bgr, input_size)
        depth = forward_depth(model, x, train_head=False)
        s = sample_points(depth, xy, ohw, input_size).cpu().numpy()
        rows.append((fn, s, ranks, bright))
    return rows


def mean_exact(rows):
    return float(np.mean([ordinal_metrics(s, r, b)[0] for (_, s, r, b) in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot_dav2")
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--input_size", type=int, default=518)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--adv", type=float, default=3.0)   # brightness-adversarial pair up-weight
    ap.add_argument("--anchor_lambda", type=float, default=0.0,
                    help=">0 adds a DENSE L2 anchor to the frozen base's (z-scored) dense output, so the "
                         "head-only sparse-point fine-tune cannot forget dense geometry. 0 = original §13.")
    ap.add_argument("--crop", type=int, default=0,
                    help="1 = center-crop(518) input matching the C3VD eval; drops point-loss imgs from train/val.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    global CROP; CROP = bool(args.crop)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    OUT = os.path.join(P1, "results/finetune", args.run); os.makedirs(OUT, exist_ok=True)
    rnd = FOLDS["rounds"][args.round]
    train, val, test = rnd["train"], rnd["val"], rnd["test"]
    if CROP:
        import crop_kvasir as CK
        keep = lambda L: [f for f in L if CK.cropped_load(IMG_ROOT, f, GTD[f])[4]]
        n0t, n0v = len(train), len(val); train, val = keep(train), keep(val)
        print(f"[crop] dropped point-loss imgs: train {n0t}->{len(train)} val {n0v}->{len(val)}", flush=True)
    print(f"[{args.run}] round {args.round} | train {len(train)} val {len(val)} test {len(test)} | crop={CROP} | dev {DEV}", flush=True)

    base = load_model()
    base_val = mean_exact(score_set(base, val, args.input_size))
    print(f"BASE val ExactMatch = {base_val:.4f}", flush=True)

    # train head only
    model = load_model()
    model.depth_head.train()
    opt = torch.optim.AdamW(model.depth_head.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val, best_state, bad = -1, None, 0
    order = list(train)
    for ep in range(args.epochs):
        model.depth_head.train()
        np.random.shuffle(order)
        tot = 0.0
        for fn in order:
            bgr, xy, ranks, bright, ohw = load_item(fn)
            x, _ = _img2tensor(bgr, args.input_size)
            depth = forward_depth(model, x, train_head=True)
            s = sample_points(depth, xy, ohw, args.input_size)
            loss = diw_ranking_loss(s, ranks, brightness=bright, adv=args.adv)
            if args.anchor_lambda > 0:                       # dense anchor: keep dense geometry close to base
                with torch.no_grad():
                    base_depth = forward_depth(base, x, train_head=False)
                zb = (base_depth - base_depth.mean()) / (base_depth.std() + 1e-6)
                zd = (depth - depth.mean()) / (depth.std() + 1e-6)
                loss = loss + args.anchor_lambda * F.mse_loss(zd, zb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        vex = mean_exact(score_set(model, val, args.input_size))
        print(f"  ep {ep:02d} loss {tot/len(order):.4f}  val_exact {vex:.4f}", flush=True)
        if vex > best_val:
            best_val = vex; best_state = {k: v.detach().cpu().clone() for k, v in model.depth_head.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop at ep {ep} (best val {best_val:.4f})", flush=True); break
    torch.save(best_state, os.path.join(OUT, "head_best.pt"))

    # evaluate base vs finetuned on the held-out TEST fold
    ft = load_model(head_state=best_state)
    base_rows = score_set(base, test, args.input_size)
    ft_rows = score_set(ft, test, args.input_size)
    with open(os.path.join(OUT, "test_points.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["filename", "point_idx", "gt_rank", "brightness", "base_score", "ft_score"])
        for (fn, sb, r, b), (_, sf, _, _) in zip(base_rows, ft_rows):
            for k in range(5):
                w.writerow([fn, k, r[k], b[k], sb[k], sf[k]])

    def agg(rows):
        em = [ordinal_metrics(s, r, b)[0] for (_, s, r, b) in rows]
        pw = [ordinal_metrics(s, r, b)[1] for (_, s, r, b) in rows]
        dd = [ordinal_metrics(s, r, b)[2] for (_, s, r, b) in rows]
        dd = [x for x in dd if not np.isnan(x)]
        return float(np.mean(em)), float(np.mean(pw)), float(np.mean(dd))
    b_em, b_pw, b_dd = agg(base_rows)
    f_em, f_pw, f_dd = agg(ft_rows)
    summary = {"run": args.run, "round": args.round, "n_test": len(test), "best_val_exact": best_val,
               "base": {"exact": b_em, "pairwise": b_pw, "discordant": b_dd},
               "finetuned": {"exact": f_em, "pairwise": f_pw, "discordant": f_dd},
               "delta": {"exact": f_em - b_em, "pairwise": f_pw - b_pw, "discordant": f_dd - b_dd}}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print("=== TEST (held-out fold) ===", flush=True)
    print(f"  base      exact {b_em:.3f} pairwise {b_pw:.3f} discordant {b_dd:.3f}", flush=True)
    print(f"  finetuned exact {f_em:.3f} pairwise {f_pw:.3f} discordant {f_dd:.3f}", flush=True)
    print(f"  DELTA     exact {f_em-b_em:+.3f} pairwise {f_pw-b_pw:+.3f} discordant {f_dd-b_dd:+.3f}", flush=True)


if __name__ == "__main__":
    main()
