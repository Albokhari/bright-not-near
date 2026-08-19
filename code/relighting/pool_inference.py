"""§16 pool_inference.py — model → score_relit adapters for the counterfactual-relight probe (E1/N4/E3-adv).

Each adapter takes a RELIT BGR uint8 image + the 5 point coords and returns nearness[5] (larger = nearer),
running the model's EXACT preprocessing so relighting happens BEFORE any model code (no model edits).
Reuses the finetune_dav2 / finetune_endoomni inference primitives. EndoOmni is imported lazily so the
DAV2-only path still works if the EndoOmni repo import is unavailable.
"""
import os, sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "finetune"))
sys.path.insert(0, os.path.join(HERE, "..", "validation"))
import finetune_dav2 as FD                                   # noqa: E402

P1 = FD.P1
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class DAVAdapter:
    """Base DAV2 (fold_heads=None) or §13/N4 cured DAV2 (per-fold OOF head). near = +depth (larger = nearer).

    Memory-frugal: the DINOv2 encoder is frozen + identical across folds, so we keep ONE DepthAnythingV2 in
    memory and hot-swap only the small DPT depth-head state per fold (avoids holding 5+ full ViT-L copies)."""

    def __init__(self, name, fold_heads=None, input_size=518):
        self.name = name
        self.fold_heads = fold_heads        # dict {fold:int -> head state_dict} for §13/N4, else None
        self.input_size = input_size
        self._model = None
        self._loaded = "base"               # which head is currently in the model

    def _get(self, fold):
        if self._model is None:
            self._model = FD.load_model()                       # base encoder + base head
        if self.fold_heads is not None:
            key = fold if fold in self.fold_heads else next(iter(self.fold_heads))
            if self._loaded != key:
                self._model.depth_head.load_state_dict(self.fold_heads[key]); self._loaded = key
        return self._model

    @torch.no_grad()
    def score_relit(self, relit_bgr, xy, ohw, fold=None):
        m = self._get(fold)
        x, _ = FD._img2tensor(relit_bgr, self.input_size)
        depth = FD.forward_depth(m, x, train_head=False)
        return FD.sample_points(depth, xy, ohw, self.input_size).cpu().numpy()


class EndoOmniAdapter:
    """EndoOmni foundation model. near = +disparity (auto-verified +1, same convention as DAV2)."""

    def __init__(self, name="EndoOmni", near_sign=1.0):
        self.name = name
        self.near_sign = near_sign
        self._model = None
        self._FE = None

    def _fe(self):
        if self._FE is None:
            import finetune_endoomni as FE                   # lazy (pulls in the EndoOmni repo)
            self._FE = FE
        return self._FE

    def _get_model(self):
        if self._model is None:
            self._model = self._fe().load_model()
        return self._model

    def _input(self, rgb01):
        FE = self._fe()
        x_np = (rgb01 * 255).astype(np.uint8)
        try:
            bp = FE.get_black_border(x_np)
            top, bottom, left, right = bp.top, bp.bottom, bp.left, bp.right
            if bottom - top < 8 or right - left < 8:
                top, bottom, left, right = 0, rgb01.shape[0], 0, rgb01.shape[1]
        except Exception:
            top, bottom, left, right = 0, rgb01.shape[0], 0, rgb01.shape[1]
        rgb_c = rgb01[top:bottom, left:right, :]
        t = torch.from_numpy(rgb_c.transpose(2, 0, 1)).contiguous().float()
        t = (t - FE._MEAN) / FE._STD
        t = F.interpolate(t.unsqueeze(0), size=(FE.IM_H, FE.IM_W), mode="bilinear",
                          align_corners=False, antialias=True)
        return t.to(DEV)

    @torch.no_grad()
    def score_relit(self, relit_bgr, xy, ohw, fold=None):
        FE = self._fe()
        rgb01 = cv2.cvtColor(relit_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = self._input(rgb01)
        disp = FE.forward_disp(self._get_model(), x, train_head=False)
        s = FE.sample_points(disp, xy, ohw, crop=None).cpu().numpy()
        return self.near_sign * s


def load_dav2_fold_heads(run="dav2", nfolds=5):
    """Load §13 per-fold head_best.pt → {fold: state_dict} for out-of-fold cured-DAV2 scoring."""
    heads = {}
    for k in range(nfolds):
        p = os.path.join(P1, "results/finetune", f"{run}_fold{k}", "head_best.pt")
        if os.path.exists(p):
            heads[k] = torch.load(p, map_location="cpu")
    return heads


def load_single_head(run):
    """Load one head_best.pt as a fold-agnostic {0: state_dict} (for the label-free dense de-confound head,
    which is a single model applied to all images — no OOF folds)."""
    p = os.path.join(P1, "results/finetune", run, "head_best.pt")
    return {0: torch.load(p, map_location="cpu")} if os.path.exists(p) else {}


def fold_of(folds, fn):
    """Return the test fold index that holds image fn (its OOF model), else None."""
    for k, rnd in enumerate(folds["rounds"]):
        if fn in rnd["test"]:
            return k
    return None
