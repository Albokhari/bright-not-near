"""DIW/WHDR-style pairwise ranking loss for sparse ordinal depth supervision.

Convention: `scores` are per-point NEARNESS scores (larger = nearer). For an
ordered pair (i, j) where point i is nearer than j (gt_rank_i < gt_rank_j), we
want score_i > score_j, so the per-pair loss is softplus(score_j - score_i) =
log(1 + exp(score_j - score_i)). Brightness-adversarial pairs (where brightness
order disagrees with GT order) can be up-weighted to directly fight the shortcut.
"""
import numpy as np
import torch


def softplus(x):
    return torch.nn.functional.softplus(x)


def pair_list(ranks):
    """All ordered index pairs (i, j) with ranks[i] < ranks[j] (i nearer)."""
    n = len(ranks)
    pairs = []
    for i in range(n):
        for j in range(n):
            if ranks[i] < ranks[j]:
                pairs.append((i, j))
    return pairs


def adversarial_weights(ranks, brightness, base=1.0, adv=3.0):
    """Per-pair weight: `adv` if the pair is brightness-adversarial (brighter point
    is the FARTHER one), else `base`. brightness: larger = brighter."""
    w = {}
    for (i, j) in pair_list(ranks):           # i nearer than j
        # brightness predicts i nearer iff brightness[i] > brightness[j].
        # adversarial = brightness says the opposite (nearer point i is darker).
        w[(i, j)] = adv if brightness[i] < brightness[j] else base
    return w


def diw_ranking_loss(scores, ranks, brightness=None, adv=1.0, reduction="mean"):
    """scores: (5,) tensor (nearness, larger=nearer). ranks: list/array len 5.
    brightness: optional (5,) array for adversarial weighting; adv>1 emphasises
    brightness-adversarial pairs (adv=1 => uniform). Returns scalar loss."""
    pairs = pair_list(ranks)
    if not pairs:
        return scores.sum() * 0.0
    if brightness is not None and adv != 1.0:
        wd = adversarial_weights(ranks, brightness, base=1.0, adv=adv)
    else:
        wd = {p: 1.0 for p in pairs}
    terms = []
    wsum = 0.0
    for (i, j) in pairs:
        w = wd[(i, j)]
        terms.append(w * softplus(scores[j] - scores[i]))
        wsum += w
    total = torch.stack(terms).sum()
    if reduction == "mean":
        return total / wsum
    return total


def rex_ranking_loss(scores, ranks, brightness, beta=1.0):
    """APPENDIX M10 — cue-environment invariant (REx/IRM-style) ordinal loss. Split GT-ordered pairs into two
    brightness ENVIRONMENTS (concordant: nearer point brighter; adversarial: nearer point darker) and penalise
    the VARIANCE of the ranking risk across environments, so the ordinal rule is stable to the shortcut cue.
      L = mean(R_concordant, R_adversarial) + beta * (R_concordant - R_adversarial)^2
    Falls back to the plain risk when one environment is empty (many images have no adversarial pair)."""
    conc, adv = [], []
    for (i, j) in pair_list(ranks):                         # i nearer than j (GT)
        term = softplus(scores[j] - scores[i])
        (adv if brightness[i] < brightness[j] else conc).append(term)   # adversarial = nearer point darker
    if conc and adv:
        Rc = torch.stack(conc).mean(); Ra = torch.stack(adv).mean()
        return 0.5 * (Rc + Ra) + beta * (Rc - Ra) ** 2
    allt = conc + adv
    return torch.stack(allt).mean() if allt else scores.sum() * 0.0


def plackett_luce_loss(scores, ranks, temperature=1.0):
    """Listwise Plackett-Luce NLL over the full ranked list (uses the WHOLE 5-point ranking, not independent
    pairs). scores: (K,) nearness (larger=nearer). ranks: len-K, 1=nearest. temperature<1 => sharper target
    (use for unanimous-consensus images). NLL = -sum_k [ s_(k) - logsumexp(s_(k:)) ] over the GT near->far order."""
    order = list(np.argsort(np.asarray(ranks)))     # indices nearest -> farthest (GT)
    s = scores / max(temperature, 1e-3)
    nll = 0.0
    for k in range(len(order)):
        rest = torch.stack([s[order[t]] for t in range(k, len(order))])
        nll = nll + (torch.logsumexp(rest, 0) - s[order[k]])
    return nll / len(order)


if __name__ == "__main__":
    # unit tests
    import numpy as np
    ranks = [1, 2, 3, 4, 5]
    # perfect prediction: nearness decreasing with rank -> loss should be small
    good = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], requires_grad=True)
    bad = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)  # fully inverted
    lg = diw_ranking_loss(good, ranks)
    lb = diw_ranking_loss(bad, ranks)
    print(f"loss(correct order)={lg.item():.4f}  loss(inverted)={lb.item():.4f}")
    assert lg.item() < lb.item(), "correct ordering must have lower loss"
    lg.backward()
    print("grad ok, shape", good.grad.shape)
    # adversarial weighting: a discordant pair gets up-weighted
    bright = np.array([0.1, 0.2, 0.9, 0.4, 0.5])  # point0 (nearest) is darkest -> adversarial
    lu = diw_ranking_loss(bad, ranks, brightness=bright, adv=1.0).item()
    la = diw_ranking_loss(bad, ranks, brightness=bright, adv=5.0).item()
    print(f"uniform={lu:.4f}  adv-weighted={la:.4f}  (adv up-weights discordant pairs)")
    print("ALL TESTS PASS")
