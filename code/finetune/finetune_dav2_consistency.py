"""§16 N4 — brightness-CONSISTENCY fine-tune (the relight closes the loop, super-novel cure).

Identical to the §13 head-only DAV2 fine-tune (finetune_dav2.py), PLUS a consistency term: under K random
per-point relights (relight.py mode='random' — brightness changes, geometry does NOT), the model's 5-point
nearness ranking must stay invariant. This directly penalises the bright=near shortcut at training time.

  L = diw_ranking_loss(s_clean, ranks, bright, adv)                       # the UNCHANGED §13 loss
      + λ · mean_k  MSE( zscore(s_relit_k) , zscore(s_clean).detach() )   # relight-invariance

λ=0 reproduces §13 exactly (ablation; the §13 dav2_fold* heads already cover it). Encoder frozen → the K
extra forwards are cheap (head-only backward). Evaluated like §13 (base vs ft on the held-out test fold:
ExactMatch / pairwise / discordant) and, downstream, by E1 reliance (must drop vs base; ideally ≤ §13).

Run (GPU):  python3 finetune_dav2_consistency.py --run dav2cons_fold0 --round 0 --lam_cons 1.0 --kviews 2
"""
import os, sys, json, csv, argparse
import numpy as np
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "counterfactual"))
import finetune_dav2 as FD
from ranking_loss import diw_ranking_loss
import relight as RL

P1, IMG_ROOT, GTD, FOLDS, DEV = FD.P1, FD.IMG_ROOT, FD.GTD, FD.FOLDS, FD.DEV


def _z(t):
    return (t - t.mean()) / (t.std() + 1e-6)


def consistency_loss(model, bgr, xy, ranks, ohw, s_clean, input_size, kviews, amp, seed0):
    """Mean MSE of z-scored relit nearness vs z-scored clean (detached) over K random relights."""
    anchor = _z(s_clean).detach()
    tot = 0.0
    for k in range(kviews):
        relit = RL.relit(bgr, xy, "random", amp=amp, seed=seed0 + k)
        x, _ = FD._img2tensor(relit, input_size)
        depth = FD.forward_depth(model, x, train_head=True)
        s_r = FD.sample_points(depth, xy, ohw, input_size)
        tot = tot + torch.mean((_z(s_r) - anchor) ** 2)
    return tot / max(kviews, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="dav2cons_fold0")
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--input_size", type=int, default=518)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--adv", type=float, default=3.0)
    ap.add_argument("--lam_cons", type=float, default=1.0)
    ap.add_argument("--kviews", type=int, default=2)
    ap.add_argument("--amp", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    OUT = os.path.join(P1, "results/finetune", args.run); os.makedirs(OUT, exist_ok=True)
    rnd = FOLDS["rounds"][args.round]
    train, val, test = rnd["train"], rnd["val"], rnd["test"]
    print(f"[{args.run}] round {args.round} | train {len(train)} val {len(val)} test {len(test)} | "
          f"λ_cons={args.lam_cons} K={args.kviews} | dev {DEV}", flush=True)

    base = FD.load_model()
    base_val = FD.mean_exact(FD.score_set(base, val, args.input_size))
    print(f"BASE val ExactMatch = {base_val:.4f}", flush=True)

    model = FD.load_model()
    model.depth_head.train()
    opt = torch.optim.AdamW(model.depth_head.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val, best_state, bad = -1, None, 0
    order = list(train)
    for ep in range(args.epochs):
        model.depth_head.train()
        np.random.shuffle(order)
        tot = tot_r = tot_c = 0.0
        for it, fn in enumerate(order):
            bgr, xy, ranks, bright, ohw = FD.load_item(fn)
            x, _ = FD._img2tensor(bgr, args.input_size)
            depth = FD.forward_depth(model, x, train_head=True)
            s = FD.sample_points(depth, xy, ohw, args.input_size)
            l_rank = diw_ranking_loss(s, ranks, brightness=bright, adv=args.adv)
            l_cons = consistency_loss(model, bgr, xy, ranks, ohw, s, args.input_size,
                                      args.kviews, args.amp, seed0=1000 * ep + it) if args.lam_cons > 0 else s.sum() * 0
            loss = l_rank + args.lam_cons * l_cons
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); tot_r += float(l_rank); tot_c += float(l_cons)
        vex = FD.mean_exact(FD.score_set(model, val, args.input_size))
        print(f"  ep {ep:02d} loss {tot/len(order):.4f} (rank {tot_r/len(order):.4f} cons {tot_c/len(order):.4f})"
              f"  val_exact {vex:.4f}", flush=True)
        if vex > best_val:
            best_val = vex; best_state = {k: v.detach().cpu().clone() for k, v in model.depth_head.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop at ep {ep} (best val {best_val:.4f})", flush=True); break
    torch.save(best_state, os.path.join(OUT, "head_best.pt"))

    ft = FD.load_model(head_state=best_state)
    base_rows = FD.score_set(base, test, args.input_size)
    ft_rows = FD.score_set(ft, test, args.input_size)
    with open(os.path.join(OUT, "test_points.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["filename", "point_idx", "gt_rank", "brightness", "base_score", "ft_score"])
        for (fn, sb, r, b), (_, sf, _, _) in zip(base_rows, ft_rows):
            for k in range(5):
                w.writerow([fn, k, r[k], b[k], sb[k], sf[k]])

    def agg(rows):
        em = [FD.ordinal_metrics(s, r, b)[0] for (_, s, r, b) in rows]
        pw = [FD.ordinal_metrics(s, r, b)[1] for (_, s, r, b) in rows]
        dd = [FD.ordinal_metrics(s, r, b)[2] for (_, s, r, b) in rows]
        dd = [x for x in dd if not np.isnan(x)]
        return float(np.mean(em)), float(np.mean(pw)), float(np.mean(dd))
    b_em, b_pw, b_dd = agg(base_rows)
    f_em, f_pw, f_dd = agg(ft_rows)
    summary = {"run": args.run, "round": args.round, "n_test": len(test), "best_val_exact": best_val,
               "lam_cons": args.lam_cons, "kviews": args.kviews,
               "base": {"exact": b_em, "pairwise": b_pw, "discordant": b_dd},
               "ft": {"exact": f_em, "pairwise": f_pw, "discordant": f_dd},
               "delta": {"exact": f_em - b_em, "pairwise": f_pw - b_pw, "discordant": f_dd - b_dd}}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(f"\n[{args.run}] base EM {b_em:.3f} disc {b_dd:.3f} -> ft EM {f_em:.3f} disc {f_dd:.3f} "
          f"(ΔEM {f_em-b_em:+.3f} Δdisc {f_dd-b_dd:+.3f})", flush=True)


if __name__ == "__main__":
    main()
