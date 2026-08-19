"""Fallback: evaluate a saved head_best.pt on the 307 benchmark (base vs ft) and write summary.json.
Use if a deconf training job is wall-killed after saving head_best.pt but before its own final eval.

  python3 eval_head.py --run deconf_ss_n3000
"""
import os, sys, json, csv, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
import finetune_dav2 as FD


def agg(rows):
    em = [FD.ordinal_metrics(s, r, b)[0] for (_, s, r, b) in rows]
    pw = [FD.ordinal_metrics(s, r, b)[1] for (_, s, r, b) in rows]
    dd = [x for x in (FD.ordinal_metrics(s, r, b)[2] for (_, s, r, b) in rows) if not np.isnan(x)]
    return float(np.mean(em)), float(np.mean(pw)), float(np.mean(dd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--input_size", type=int, default=518)
    args = ap.parse_args()
    OUT = os.path.join(FD.P1, "results/finetune", args.run)
    files = list(FD.GTD.keys())
    base = FD.load_model()
    full = os.path.join(OUT, "model_best.pt")
    if os.path.exists(full):                       # encoder-unfreeze run: full model state
        ft = FD.load_model(); ft.load_state_dict(torch.load(full, map_location="cpu")); ft = ft.to(FD.DEV).eval()
    else:
        ft = FD.load_model(head_state=torch.load(os.path.join(OUT, "head_best.pt"), map_location="cpu"))
    b = agg(FD.score_set(base, files, args.input_size))
    f = agg(FD.score_set(ft, files, args.input_size))
    summary = {"run": args.run, "mode": "ss", "n_eval": len(files),
               "base": {"exact": b[0], "pairwise": b[1], "discordant": b[2]},
               "ft": {"exact": f[0], "pairwise": f[1], "discordant": f[2]},
               "delta": {"exact": f[0] - b[0], "pairwise": f[1] - b[1], "discordant": f[2] - b[2]}}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(f"[{args.run}] base EM {b[0]:.3f} disc {b[2]:.3f} -> ft EM {f[0]:.3f} disc {f[2]:.3f} "
          f"(dEM {f[0]-b[0]:+.3f} dDisc {f[2]-b[2]:+.3f} dPair {f[1]-b[1]:+.3f})")


if __name__ == "__main__":
    main()
