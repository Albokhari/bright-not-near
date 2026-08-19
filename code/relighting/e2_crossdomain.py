"""§16 E2 — controlled C3VD-vs-real fine-tune (the rigor centerpiece for finding #1).

The internally-controlled twin of the EndoSfM3D anecdote: take the SAME DAV2 ViT-L and the EXACT §13 head-only
ordinal recipe, change ONLY the training DOMAIN — phantom C3VD ordinal labels (make_c3vd_ordinal.py) vs the
real-tissue Kvasir labels (§13) — and cross-evaluate BOTH on the 307 real-tissue ordinal benchmark.

The real-tissue eval column is LEAKAGE-FREE (real tissue is never in C3VD training), so the headline 2×1
comparison needs no fold gymnastics: a single C3VD-trained head scored on all 307 real images is out-of-domain
by construction. Compared against {base DAV2, §13 real-trained (OOF)} already on disk.
Claim: training on phantom geometry does NOT teach real-tissue ordinal geometry (it may even hurt), while
training on real tissue does — same architecture, same loss, only the domain differs.

Phase A (GPU, train on C3VD): swap FD data roots → train head on C3VD ordinal → save head.
Phase B (eval on real):       restore roots → score base / C3VD-head / §13-OOF on all 307 → metrics + CI.
Run: python3 e2_crossdomain.py --epochs 40
"""
import os, sys, json, argparse
import numpy as np
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "finetune"))
sys.path.insert(0, os.path.join(HERE, "..", "validation"))
import finetune_dav2 as FD
from ranking_loss import diw_ranking_loss
import pool_inference as PI
import p1_common as C

P1 = FD.P1
C3_IMG = "/well/rittscher/users/fxh757/Datasets/PUBLIC_ENDO/Depth_split_2/test/images"
OUTDIR = os.path.join(P1, "results/counterfactual")
KVASIR_IMG = FD.IMG_ROOT
KVASIR_GTD = dict(FD.GTD)                                    # snapshot the real-tissue GT before swapping


def train_on_c3vd(args):
    gt = json.load(open(os.path.join(OUTDIR, "c3vd_ordinal_gt.json")))
    split = json.load(open(os.path.join(OUTDIR, "c3vd_ordinal_split.json")))
    FD.IMG_ROOT = C3_IMG
    FD.GTD = {im["filename"]: im for im in gt["images"]}
    train, val = split["train"], split["val"]
    print(f"[E2] train DAV2 head on C3VD ordinal: train {len(train)} val {len(val)} | dev {FD.DEV}", flush=True)

    model = FD.load_model(); model.depth_head.train()
    opt = torch.optim.AdamW(model.depth_head.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val, best_state, bad = -1, None, 0
    order = list(train)
    for ep in range(args.epochs):
        model.depth_head.train(); np.random.shuffle(order); tot = 0.0
        for fn in order:
            bgr, xy, ranks, bright, ohw = FD.load_item(fn)
            x, _ = FD._img2tensor(bgr, args.input_size)
            depth = FD.forward_depth(model, x, train_head=True)
            s = FD.sample_points(depth, xy, ohw, args.input_size)
            loss = diw_ranking_loss(s, ranks, brightness=bright, adv=args.adv)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        vex = FD.mean_exact(FD.score_set(model, val, args.input_size))
        print(f"  ep {ep:02d} loss {tot/len(order):.4f} c3vd_val_exact {vex:.4f}", flush=True)
        if vex > best_val:
            best_val = vex; best_state = {k: v.detach().cpu().clone() for k, v in model.depth_head.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop ep {ep} (best c3vd val {best_val:.4f})", flush=True); break
    run = os.path.join(P1, "results/finetune", "c3vd_ordinal_head"); os.makedirs(run, exist_ok=True)
    torch.save(best_state, os.path.join(run, "head_best.pt"))
    return best_state, best_val


def eval_on_real(c3vd_state, args):
    """Score base / C3VD-trained / §13-real-OOF on all 307 real images."""
    FD.IMG_ROOT = KVASIR_IMG; FD.GTD = KVASIR_GTD
    files = sorted(KVASIR_GTD.keys())
    folds = FD.FOLDS
    base = FD.load_model()
    c3vd = FD.load_model(head_state=c3vd_state)
    s13_heads = PI.load_dav2_fold_heads("dav2")
    s13_models = {k: FD.load_model(head_state=h) for k, h in s13_heads.items()}

    def metrics_for(score_fn):
        em, pw, dd = [], [], []
        for fn in files:
            bgr, xy, ranks, bright, ohw = FD.load_item(fn)
            s = score_fn(fn, bgr, xy, ohw)
            e, p, d = FD.ordinal_metrics(s, ranks, bright)[:3]
            em.append(e); pw.append(p); dd.append(d)
        return np.array(em, float), np.array(pw, float), np.array(dd, float)

    @torch.no_grad()
    def sc(model, bgr, xy, ohw):
        x, _ = FD._img2tensor(bgr, args.input_size)
        depth = FD.forward_depth(model, x, train_head=False)
        return FD.sample_points(depth, xy, ohw, args.input_size).cpu().numpy()

    rows = {}
    rows["base"] = metrics_for(lambda fn, b, xy, ohw: sc(base, b, xy, ohw))
    rows["c3vd_trained"] = metrics_for(lambda fn, b, xy, ohw: sc(c3vd, b, xy, ohw))
    rows["real_trained_s13_oof"] = metrics_for(
        lambda fn, b, xy, ohw: sc(s13_models[PI.fold_of(folds, fn)], b, xy, ohw))

    out = {}
    print("\n=== E2 cross-domain — evaluated on 307 REAL-tissue ordinal ===")
    print(f"{'arm':<22}{'ExactMatch[CI]':>22}{'pairwise':>10}{'discordant[CI]':>22}")
    for k, (em, pw, dd) in rows.items():
        e_m, e_lo, e_hi = C.cluster_bootstrap_ci(em)
        d_m, d_lo, d_hi = C.cluster_bootstrap_ci(dd)
        out[k] = {"exact": [e_m, e_lo, e_hi], "pairwise": float(np.nanmean(pw)), "discordant": [d_m, d_lo, d_hi]}
        print(f"{k:<22}{e_m:>7.3f}[{e_lo:.3f},{e_hi:.3f}]{np.nanmean(pw):>10.3f}{d_m:>9.3f}[{d_lo:.3f},{d_hi:.3f}]")
    # paired Δ vs base, image-clustered
    for k in ("c3vd_trained", "real_trained_s13_oof"):
        d_em = rows[k][0] - rows["base"][0]; d_dd = rows[k][2] - rows["base"][2]
        em_m, em_lo, em_hi = C.cluster_bootstrap_ci(d_em); dd_m, dd_lo, dd_hi = C.cluster_bootstrap_ci(d_dd)
        out[k]["dExact_vs_base"] = [em_m, em_lo, em_hi]; out[k]["dDisc_vs_base"] = [dd_m, dd_lo, dd_hi]
        print(f"  Δ {k} vs base: ΔEM {em_m:+.3f}[{em_lo:+.3f},{em_hi:+.3f}]  Δdisc {dd_m:+.3f}[{dd_lo:+.3f},{dd_hi:+.3f}]")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_size", type=int, default=518)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--adv", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    state, cval = train_on_c3vd(args)
    out = eval_on_real(state, args)
    out["c3vd_best_val_exact"] = cval
    json.dump(out, open(os.path.join(OUTDIR, "E2_crossdomain.json"), "w"), indent=1)
    print(f"\nwrote results/counterfactual/E2_crossdomain.json")


if __name__ == "__main__":
    main()
