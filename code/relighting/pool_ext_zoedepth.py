"""§16-ext pool_ext_zoedepth.py — ZoeDepth score_relit adapter for the metric-GT decorrelation probe.

Drop-in for e_phantom_decouple.py alongside pool_inference.DAVAdapter / EndoOmniAdapter. Reuses the model's
loading + preprocessing EXACTLY from experiments/external_baselines_newmodels/run_zoedepth.py:
  - the 3 timm-drift compat monkeypatches (run_zoedepth._apply_compat_patches),
  - torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True, trust_repo=True) then strict-load restore,
  - RGB-PIL -> model.infer_pil(pil) inference (HxW METRIC depth in meters, larger = FARTHER).

Orientation: ZoeDepth emits metric DISTANCE (larger = farther), so we INVERT (negate) the sampled depth to
produce NEARNESS (larger = nearer), matching DAVAdapter's convention. Non-finite depths are set to the frame's
finite max (farthest, exactly as run_zoedepth.py does) so they map to the most-negative nearness.
"""
import os, sys
import numpy as np
import cv2
import torch
from PIL import Image

HERE = os.path.dirname(__file__)
# pool_inference gives us DEV + the shared bilinear point sampler (PI.FD.sample_points), same as DAVAdapter.
sys.path.insert(0, HERE)
import pool_inference as PI                                                          # noqa: E402
# the ZoeDepth run script (loading + preprocessing + orientation, reused verbatim).
EXT = os.path.abspath(os.path.join(HERE, "..", "..", "experiments", "external_baselines_newmodels"))
sys.path.insert(0, EXT)
import run_zoedepth as RZD                                                           # noqa: E402


class ZoeDepthAdapter:
    """ZoeDepth (ZoeD_N, BEiT384-L MiDaS backbone). near = -metric_depth (invert distance -> nearness).

    The model is heavy (ViT-L) so it is loaded ONCE on first use and cached to PI.DEV."""

    def __init__(self, name="ZoeDepth", model_type="ZoeD_N"):
        self.name = name
        self.model_type = model_type
        self._model = None

    def _get(self):
        if self._model is None:
            # --- reuse run_zoedepth.py's loading EXACTLY -------------------------------------------
            _origload = RZD._apply_compat_patches()                 # non-strict load + int-size + drop_path alias
            model = torch.hub.load("isl-org/ZoeDepth", self.model_type,
                                   pretrained=True, trust_repo=True)
            from torch.nn.modules.module import Module              # restore strict load (build is complete)
            Module.load_state_dict = _origload
            self._model = model.to(PI.DEV).eval()
        return self._model

    @torch.no_grad()
    def score_relit(self, relit_bgr, xy, ohw, fold=None):
        m = self._get()
        # preprocessing EXACTLY as run_zoedepth.py: RGB PIL -> infer_pil (metric depth, meters, larger=farther)
        rgb = cv2.cvtColor(relit_bgr, cv2.COLOR_BGR2RGB)
        depth = m.infer_pil(Image.fromarray(rgb))                  # HxW metric depth at ORIGINAL input resolution
        depth = np.asarray(depth, dtype=np.float32)
        # sanitize non-finite -> finite max (farthest), matching run_zoedepth.py's write path.
        finite = np.isfinite(depth)
        if finite.any():
            depth = np.where(finite, depth, depth[finite].max())
        else:
            depth = np.zeros_like(depth)
        d = torch.from_numpy(np.ascontiguousarray(depth)).to(PI.DEV)
        s = PI.FD.sample_points(d, xy, ohw, self.model_type).cpu().numpy()  # metric distance at the 5 pts
        return -s                                                  # INVERT: distance -> nearness (larger = nearer)


if __name__ == "__main__":
    # Smoke test: build the model and forward a dummy 256x256 image so load/checkpoint errors surface early.
    ad = ZoeDepthAdapter("ZoeDepth")
    m = ad._get()
    dummy_bgr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(dummy_bgr, cv2.COLOR_BGR2RGB)
    depth = np.asarray(m.infer_pil(Image.fromarray(rgb)), dtype=np.float32)
    print("model class      :", type(m).__name__)
    print("device           :", PI.DEV)
    print("depth out shape  :", depth.shape, "dtype", depth.dtype,
          "range %.3f..%.3f" % (float(np.nanmin(depth)), float(np.nanmax(depth))))
    xy = np.array([[10, 10], [50, 50], [128, 128], [200, 60], [240, 240]], dtype=float)
    near = ad.score_relit(dummy_bgr, xy, (256, 256))
    print("nearness[5]      :", np.asarray(near))
    print("SMOKE_OK")
