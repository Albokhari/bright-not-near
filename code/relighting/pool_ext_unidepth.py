"""§16-ext UniDepthAdapter — score_relit adapter for the metric-GT decorrelation probe (e_phantom_decouple.py).

Wraps UniDepth-V2 (lpiccinelli-eth/UniDepth) with the SAME interface as pool_inference.DAVAdapter /
EndoOmniAdapter so the counterfactual-relight probe can score UniDepth on RELIT C3VD images without
touching model code (relighting happens in the BGR image BEFORE any model preprocessing).

Loading + preprocessing + camera + depth orientation are copied EXACTLY from the model's run script
experiments/external_baselines_newmodels/run_unidepth.py:
  - UniDepthV2.from_pretrained(repo_id); model.interpolation_mode = "bilinear"; .to(DEV).eval()
  - input: RGB uint8 CHW tensor (infer() std-izes internally); canonical pinhole K (fx=fy=W, ppt=center)
  - output: pred["depth"] = METRIC depth in meters (distance: LARGER = FARTHER, near = smaller)

UniDepth outputs metric DISTANCE, so we INVERT (negate) to get nearness, matching DAVAdapter's
convention (LARGER = NEARER). Negation is a strict order-reversing map, so it preserves the point
ordering the probe scores (RZ.pairwise_acc is rank-based) with no divide-by-zero risk.
"""
import os, sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
import pool_inference as PI                                    # noqa: E402  (exposes PI.DEV, PI.P1)

DEV = PI.DEV
REPO = os.path.join(PI.P1, "experiments/external_baselines_newmodels/UniDepth")
if REPO not in sys.path:
    sys.path.insert(0, REPO)


class UniDepthAdapter:
    """UniDepth-V2 metric-depth foundation model. near = -depth (invert: model outputs metric distance,
    larger = farther). Model is loaded ONCE to PI.DEV; scoring is preprocessing-faithful to run_unidepth.py."""

    def __init__(self, name="UniDepth", repo_id="lpiccinelli/unidepth-v2-vitl14"):
        self.name = name
        self.repo_id = repo_id
        from unidepth.models import UniDepthV2                 # noqa: E402 (needs REPO on sys.path)
        from unidepth.utils.camera import Pinhole
        self._Pinhole = Pinhole
        m = UniDepthV2.from_pretrained(repo_id)
        m.interpolation_mode = "bilinear"                      # match run script
        self._model = m.to(DEV).eval()

    def _camera(self, oh, ow):
        # canonical pinhole: focal ~= image width, principal point at center (relative order only)
        K = torch.tensor([[float(ow), 0.0, ow / 2.0],
                          [0.0, float(ow), oh / 2.0],
                          [0.0, 0.0, 1.0]], dtype=torch.float32)
        return self._Pinhole(K=K.unsqueeze(0))

    @torch.no_grad()
    def score_relit(self, relit_bgr, xy, ohw, fold=None):
        oh, ow = ohw
        rgb = cv2.cvtColor(relit_bgr, cv2.COLOR_BGR2RGB)       # HxWx3 uint8 RGB (== run script's convert("RGB"))
        rgb_t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)   # uint8 CHW; infer() does /255 + norm
        camera = self._camera(oh, ow)
        pred = self._model.infer(rgb_t, camera)
        depth = pred["depth"].squeeze().float()               # (H,W) metric meters, larger = farther
        # replace non-finite with the finite max (== farthest), matching run_unidepth.py's fallback
        finite = torch.isfinite(depth)
        if not bool(finite.all()):
            fill = depth[finite].max() if bool(finite.any()) else torch.zeros((), device=depth.device)
            depth = torch.where(finite, depth, fill)
        # sample the 5 original-res points, DAVAdapter convention (normalise by ohw, align_corners=True)
        near = self._sample_points(depth, xy, oh, ow)          # (5,) metric distance
        return (-near).cpu().numpy()                           # INVERT: larger = nearer

    @staticmethod
    def _sample_points(depth, xy, oh, ow):
        Hh, Ww = depth.shape
        gx = (np.asarray(xy)[:, 0] / ow) * 2 - 1               # normalise to [-1,1] over full original size
        gy = (np.asarray(xy)[:, 1] / oh) * 2 - 1
        grid = torch.tensor(np.stack([gx, gy], 1), dtype=depth.dtype,
                            device=depth.device).view(1, 1, len(gx), 2)
        s = F.grid_sample(depth.view(1, 1, Hh, Ww), grid, mode="bilinear", align_corners=True)
        return s.view(-1)


if __name__ == "__main__":
    # Smoke test: build the model + one forward on a dummy 256x256 BGR image (catches load/checkpoint errors).
    print(f"[UniDepthAdapter] dev={DEV} repo={REPO}", flush=True)
    ad = UniDepthAdapter("UniDepth")
    print(f"model class: {type(ad._model).__name__}", flush=True)
    dummy = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)   # HxWx3 uint8 BGR
    xy = np.array([[40, 40], [200, 40], [128, 128], [40, 200], [200, 200]], float)
    with torch.no_grad():
        depth = ad._model.infer(
            torch.from_numpy(cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB)).permute(2, 0, 1),
            ad._camera(256, 256),
        )["depth"].squeeze().float()
    near = ad.score_relit(dummy, xy, (256, 256))
    print(f"raw depth output shape: {tuple(depth.shape)} (metric meters, larger=farther)", flush=True)
    print(f"nearness score shape: {near.shape} values(larger=nearer): {np.round(near, 4)}", flush=True)
    print("SMOKE_OK", flush=True)
