"""5-fold image-disjoint split of the 307 consensus images (rotating train/val/test CV).

Round k: test = fold k; val = first VAL_FRAC of the remaining folds; train = rest.
Every image is in test exactly once -> the full 307 benchmark is preserved as the
aggregate eval set. Deterministic (seeded). Writes folds.json.
"""
import os
import json
import numpy as np

P1 = "/well/rittscher/users/fxh757/Code/Papers/P1_Ordinal_Endo_Depth_Benchmark__EndoLINA_MICCAI2026"
GT = os.path.join(P1, "results/consensus_GT/merged_pass_consensus_gt.json")
OUT = os.path.join(P1, "results/finetune/folds.json")
K = 5
VAL_FRAC = 0.15
SEED = 0


def main():
    imgs = json.load(open(GT))["images"]
    files = sorted(im["filename"] for im in imgs)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(files))
    folds = [[] for _ in range(K)]
    for i, idx in enumerate(perm):
        folds[i % K].append(files[idx])

    rounds = []
    for k in range(K):
        test = sorted(folds[k])
        rest = [f for j in range(K) if j != k for f in folds[j]]
        rest = list(rng.permutation(rest))  # shuffle the train+val pool
        nval = int(round(VAL_FRAC * len(rest)))
        val = sorted(rest[:nval])
        train = sorted(rest[nval:])
        rounds.append({"round": k, "train": train, "val": val, "test": test,
                       "n_train": len(train), "n_val": len(val), "n_test": len(test)})

    out = {"k": K, "seed": SEED, "val_frac": VAL_FRAC, "n_total": len(files),
           "folds": folds, "rounds": rounds, "pilot_round": 0}
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")
    for r in rounds:
        print(f"  round {r['round']}: train={r['n_train']} val={r['n_val']} test={r['n_test']}")
    # sanity: disjoint test folds cover all 307
    allt = sorted(f for r in rounds for f in r["test"])
    assert allt == sorted(files), "test folds must partition the 307"
    print("OK: 5 test folds partition all", len(files), "images")


if __name__ == "__main__":
    main()
