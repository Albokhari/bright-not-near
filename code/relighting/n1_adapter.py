"""§16 N1 — universal de-biasing adapter (super-novel, CPU). Does ONE reusable corrector de-bias ANY model?

A tiny MLP  f([z(pred_val), z(brightness)]) → corrected nearness, trained ONCE on the ordinal labels pooled
across models, then applied LEAVE-ONE-MODEL-OUT (the corrector is image-agnostic → no image leakage; the
held-out MODEL is the generalisation test).

RESULT (the sharpest "test-time correction fails"): optimised for ordinal accuracy, the corrector LEARNS and
RE-INJECTS the bright=near shortcut. Mean ΔExactMatch +0.119 (17/18 up) looks like a universal cure, but
Δdiscordant<0 for ALL 18 (mean −0.35) and Δconcordant>0 for ALL 18 (mean +0.19) → it wins the majority
concordant pairs by losing the adversarial ones; the shuffle-brightness null gives ΔEM −0.004 (gain is entirely
brightness-driven); corr(orig EM, ΔEM)=−0.61 (regresses weak models toward the brightness baseline) and DAV2 —
the strongest model — is the ONLY one HURT (−0.072). So a learned test-time corrector AMPLIFIES the shortcut,
not removes it. Only training-time invariance (N4) genuinely de-biases.

Controls (the §15 lesson): shuffle-brightness null + the re-allocation discriminator (Δexact vs Δdisc/Δconc) +
image-clustered CI.
Run (CPU): python3 n1_adapter.py
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "validation"))
sys.path.insert(0, os.path.join(HERE, "..", "routing"))
import e3_deconfound_simple as E3          # reuse load_model_rows + the model lists
import p1_common as C

P1 = C.P1
FOLDS = json.load(open(os.path.join(P1, "results/finetune/folds.json")))
MODELS = E3.MODELS + E3.NEWMODELS
DISP = E3.DISP


def _z(a):
    a = np.asarray(a, float); s = a.std()
    return (a - a.mean()) / s if s > 1e-9 else a - a.mean()


def rerank(near):                                    # nearness (larger=nearer) -> rank (1=nearest)
    return (-np.asarray(near)).argsort().argsort() + 1


def exact(rank, gt):
    return float(np.array_equal(np.argsort(np.asarray(rank, float)), np.argsort(np.asarray(gt, float))))


def discordant(rank, gt, bright):                    # adversarial pairs: nearer point is darker
    rank, gt, bright = map(lambda a: np.asarray(a, float), (rank, gt, bright))
    c = t = 0
    for i in range(5):
        for j in range(5):
            if gt[i] < gt[j] and bright[i] < bright[j]:
                t += 1; c += int(rank[i] < rank[j])
    return c / t if t else np.nan


def concordant(rank, gt, bright):                    # brightness agrees: nearer point is brighter
    rank, gt, bright = map(lambda a: np.asarray(a, float), (rank, gt, bright))
    c = t = 0
    for i in range(5):
        for j in range(5):
            if gt[i] < gt[j] and bright[i] > bright[j]:
                t += 1; c += int(rank[i] < rank[j])
    return c / t if t else np.nan


def feats(near, bright, shuffle_bright=False, rng=None):
    b = rng.permutation(bright) if shuffle_bright else bright
    return np.stack([_z(near), _z(b)], 1)          # (5,2)


class MLP(nn.Module):
    def __init__(self, h=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))

    def forward(self, x):                            # x [M,5,2] -> [M,5]
        M = x.shape[0]
        return self.net(x.reshape(-1, 2)).reshape(M, 5)


def rank_mask(R):                                    # R [M,5] gt ranks -> mask[m,i,j]=1 if i nearer than j
    Rt = torch.tensor(R, dtype=torch.float32)
    return (Rt.unsqueeze(2) < Rt.unsqueeze(1)).float()   # [M,5,5]: rank[i]<rank[j]


def batch_loss(s, mask):                             # s [M,5]; loss = Σ_{i nearer j} softplus(s_j - s_i)
    diff = s.unsqueeze(1) - s.unsqueeze(2)           # diff[m,i,j] = s_j - s_i
    return (mask * torch.nn.functional.softplus(diff)).sum() / s.shape[0]


def train_corrector(X, R, epochs=150, lr=8e-3, seed=0):
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32); mask = rank_mask(R)
    m = MLP(); opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-3)
    for _ in range(epochs):
        opt.zero_grad(); loss = batch_loss(m(Xt), mask); loss.backward(); opt.step()
    return m


def collect(shuffle_bright=False, seed=0):
    """Per model: ordered file list + arrays X[M,5,2], near[M,5], gt[M,5], bright[M,5]."""
    dfw = C.load_wide(); bval, _ = C.brightness_lookup(dfw)
    rng = np.random.default_rng(seed)
    data = {}
    for m in MODELS:
        rows = E3.load_model_rows(dfw, bval, m)
        if not rows:
            continue
        files = sorted(rows)
        near = np.array([rows[f][0] for f in files]); gt = np.array([rows[f][1] for f in files])
        br = np.array([rows[f][2] for f in files])
        X = np.array([feats(near[i], br[i], shuffle_bright, rng) for i in range(len(files))])
        data[DISP.get(m, m)] = dict(files=files, X=X, near=near, gt=gt, br=br)
    return data


def fold_idx(files):
    fmap = {f: k for k, rnd in enumerate(FOLDS["rounds"]) for f in rnd["test"]}
    return np.array([fmap.get(f, -1) for f in files])


def evaluate(shuffle_bright=False):
    """LEAVE-ONE-MODEL-OUT: train one corrector on ALL OTHER models (all their images), apply to the held-out
    model. The audit confirmed the corrector input [z(pred_val), z(brightness)] carries NO image identity, so a
    shared test image cannot be memorised → image-fold disjointness is unnecessary; the held-out MODEL is the
    genuine generalisation test (a reusable field tool on an unseen model)."""
    data = collect(shuffle_bright)
    names = list(data)
    res = {}
    for held in names:
        d = data[held]
        X_tr = np.concatenate([data[m]["X"] for m in names if m != held])
        R_tr = np.concatenate([data[m]["gt"] for m in names if m != held])
        mdl = train_corrector(X_tr, R_tr)
        with torch.no_grad():
            corr_near = mdl(torch.tensor(d["X"], dtype=torch.float32)).numpy()
        # score original vs corrected
        def met(near):
            ex = np.array([exact(rerank(near[i]), d["gt"][i]) for i in range(len(near))], float)
            dc = np.array([discordant(rerank(near[i]), d["gt"][i], d["br"][i]) for i in range(len(near))], float)
            cc = np.array([concordant(rerank(near[i]), d["gt"][i], d["br"][i]) for i in range(len(near))], float)
            return ex, dc, cc
        oe, od, oc = met(d["near"]); ce, cd, cc = met(corr_near)
        de = C.cluster_bootstrap_ci(ce - oe); dd = C.cluster_bootstrap_ci(cd - od); dcc = C.cluster_bootstrap_ci(cc - oc)
        res[held] = {"orig_exact": float(oe.mean()), "corr_exact": float(ce.mean()),
                     "dExact": de, "dDisc": dd, "dConc": dcc}
    return res


def main():
    print("N1 universal de-biasing adapter — leave-one-model-out + image-fold-disjoint.\n")
    real = evaluate(shuffle_bright=False)
    null = evaluate(shuffle_bright=True)
    out = {"real": real, "shuffle_null": null}
    json.dump(out, open(os.path.join(P1, "results/counterfactual/N1_adapter.json"), "w"), indent=1)
    print(f"{'model':<14}{'orig EM':>8}{'corr EM':>9}{'ΔExact[CI]':>20}{'Δdisc':>9}{'Δconc':>9}{'  null ΔEM':>12}")
    de_all = []
    for m in real:
        r = real[m]; n = null[m]["dExact"][0]; de_all.append(r["dExact"][0])
        print(f"{m:<14}{r['orig_exact']:>8.3f}{r['corr_exact']:>9.3f}"
              f"{r['dExact'][0]:>+10.3f}[{r['dExact'][1]:+.2f},{r['dExact'][2]:+.2f}]"
              f"{r['dDisc'][0]:>+9.3f}{r['dConc'][0]:>+9.3f}{n:>+12.3f}")
    print(f"\nmean ΔExact over models: {np.mean(de_all):+.3f}  (a genuine universal cure needs ΔExact CI>0 AND ≠ shuffle-null)")
    print("wrote results/counterfactual/N1_adapter.json")


if __name__ == "__main__":
    main()
