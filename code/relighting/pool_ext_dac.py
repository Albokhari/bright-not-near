"""§16-ext pool_ext_dac.py — DepthAnything-AC → score_relit adapter for the metric-GT decorrelation
probe (code/counterfactual/e_phantom_decouple.py). Standalone companion to pool_inference.py
(DAVAdapter / EndoOmniAdapter); does NOT edit pool_inference.py.

Mirrors the DAVAdapter interface EXACTLY:
    DACAdapter(name)  ->  .score_relit(relit_bgr, xy, ohw, fold=None) -> nearness[5]  (larger = NEARER)

Model build + preprocessing + orientation are reused verbatim from the model's run script
(experiments/external_baselines_newmodels/run_depthanythingac.py), which is already wrapped by the
finetune primitive module code/finetune/finetune_depthanythingac.py:
  - load_model():   DepthAnything_AC(vits cfg); torch.load(ckpt) -> load_state_dict(strict=False);
                    eval; encoder frozen; moved to DEV. Loaded ONCE here and reused across relit
                    forwards. (Uses a chdir(REPO) so the DINOv2-ViT-S backbone's RELATIVE-path
                    checkpoint resolves — see finetune_depthanythingac.load_model.)
  - _img2tensor():  cv2 BGR->RGB /255, side min(h,w)->input_size rounded up to mult of 14 (INTER_CUBIC),
                    ImageNet mean/std normalize — identical to run_depthanythingac.preprocess_image.
  - forward_depth():model.depth_head(feats,...)["out"] — post-ReLU RELATIVE DISPARITY (larger = NEARER,
                    exactly like DAV2). So NEARNESS = +disparity (near_sign = +1); NO inversion.

Point scores use FD.sample_points (differentiable bilinear grid_sample at original-res coords),
same as DAVAdapter/EndoOmniAdapter.
"""
import os, sys

os.environ.setdefault("XFORMERS_DISABLED", "1")  # DINOv2 plain-attention fallback (matches the job env)

import numpy as np
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "finetune"))
import finetune_dav2 as FD                         # noqa: E402  model-agnostic sample_points
import finetune_depthanythingac as DAC             # noqa: E402  DAC load_model / _img2tensor / forward_depth

DEV = DAC.DEV


class DACAdapter:
    """DepthAnything-AC (ViT-S; "Depth Anything At Any Condition", arXiv 2507.01634): a DAV2-clone
    DPT-DINOv2 fine-tuned for robustness to adverse conditions. Outputs relative DISPARITY (post-ReLU),
    larger = NEARER — SAME sign convention as DAV2 — so nearness = +disparity (near_sign = +1); no
    inversion. The frozen ViT-S encoder + DPT head are loaded ONCE and reused for every relit forward."""

    def __init__(self, name="DepthAnythingAC", input_size=518, near_sign=1.0):
        self.name = name
        self.input_size = input_size
        self.near_sign = near_sign               # +1: disparity is already near=larger (matches DAVAdapter)
        self._model = DAC.load_model()           # ViT-S encoder + DPT head -> DAC.DEV, eval(), encoder frozen

    @torch.no_grad()
    def score_relit(self, relit_bgr, xy, ohw, fold=None):
        # relit_bgr: HxWx3 uint8 BGR (relighting already applied BEFORE any model code).
        x, _ = DAC._img2tensor(relit_bgr, self.input_size)            # exact run-script preprocessing -> DAC.DEV
        depth = DAC.forward_depth(self._model, x, train_head=False)   # [H',W'] relative disparity (larger=nearer)
        s = FD.sample_points(depth, xy, ohw, self.input_size).cpu().numpy()  # 5 pts, original-res int coords
        return self.near_sign * s                                     # length-5, larger = NEARER


if __name__ == "__main__":
    # Smoke test: build the model (catches ckpt/backbone load errors) + print class + output shape.
    ad = DACAdapter("DepthAnythingAC-smoke")
    print("model class      :", type(ad._model).__name__)
    dummy = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)  # HxWx3 uint8 BGR
    x, ohw = DAC._img2tensor(dummy, ad.input_size)
    depth = DAC.forward_depth(ad._model, x, train_head=False)
    print("input tensor     :", tuple(x.shape), "on", x.device)
    print("orig (h,w)       :", ohw)
    print("disparity output :", tuple(depth.shape), depth.dtype)
    xy = np.array([[10, 10], [200, 50], [128, 128], [50, 200], [240, 240]], float)  # original-res int coords
    near = ad.score_relit(dummy, xy, ohw)
    print("score_relit      :", near.shape, np.asarray(near))
    print("SMOKE_OK")
