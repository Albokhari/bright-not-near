"""§16-ext pool_ext_metric3d.py — Metric3D → score_relit adapter for the counterfactual-relight probe.

Same interface as pool_inference.py:DAVAdapter / EndoOmniAdapter: takes a RELIT BGR uint8 image + the 5
point coords and returns nearness[5] (LARGER = NEARER), running Metric3D's EXACT preprocessing so the
relighting happens BEFORE any model code (no model edits).

Loading + preprocessing + the depth→metric rescale are reused verbatim from the run script
experiments/external_baselines_newmodels/run_metric3d.py (imported as RM3D):
  transform_test_data_scalecano (canonical-camera resize using the canonical intrinsic, pad, normalize)
  -> model.inference -> de-pad, resize back to original, rescale by depth_range[1] / label_scale_factor.
We use Metric3D's default canonical intrinsic fx=fy=1000, principal point at image center — identical to
the run script (only the RELATIVE depth ORDER is needed, which is robust to the exact intrinsic).

NEARNESS orientation
--------------------
Metric3D outputs METRIC depth in METERS (distance: larger = FARTHER). DAVAdapter's convention is
larger = NEARER. We therefore INVERT by negation: nearness = -metric_depth. Negation is strictly
order-preserving (unlike a per-image renormalisation) and avoids the divide-by-zero of a reciprocal,
so the 5-point ranking matches DAVAdapter's orientation exactly. Non-finite / non-positive pixels are
set to the per-image max depth (far) before negation, mirroring run_metric3d.py's invalid→far rule.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
# reuse the exact loading/preprocessing from the run script + its DAV2 sample_points primitive
sys.path.insert(0, os.path.join(HERE, "..", "..", "experiments", "external_baselines_newmodels"))
sys.path.insert(0, HERE)
import pool_inference as PI          # noqa: E402  (gives PI.DEV + PI.FD.sample_points, same as templates)
import run_metric3d as RM3D          # noqa: E402  (module-level insert puts Metric3D/ on sys.path)


class Metric3DAdapter:
    """Metric3D ConvNeXt-Large (JUGGHM/Metric3D). near = -metric_depth (metric distance inverted → nearer)."""

    def __init__(self, name="Metric3D", ckpt=None):
        self.name = name
        self.ckpt = ckpt or RM3D.CKPT
        self.cfg = RM3D.load_config()
        self.normalize_scale = self.cfg.data_basic["depth_range"][1]  # 150

        from mono.model.monodepth_model import get_configured_monodepth_model
        model = get_configured_monodepth_model(self.cfg)
        sd = torch.load(self.ckpt, map_location="cpu")
        state = sd.get("model_state_dict", sd)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[Metric3DAdapter] loaded ckpt: {len(missing)} missing, {len(unexpected)} unexpected keys",
              flush=True)
        self._model = model.to(PI.DEV).eval()

    @torch.no_grad()
    def _infer_metric_depth(self, relit_bgr):
        """Forward Metric3D on a HxWx3 uint8 BGR image → metric depth (meters) tensor at ORIGINAL res.
        Mirrors run_metric3d.py exactly (RGB input, canonical intrinsic, de-pad, resize-back, rescale)."""
        rgb_origin = relit_bgr[:, :, ::-1]                       # BGR uint8 -> RGB (run script uses RGB)
        rgb_origin = np.ascontiguousarray(rgb_origin)
        ori_h, ori_w = rgb_origin.shape[:2]
        # canonical default intrinsic: fx=fy=1000, principal point at image center (== run_metric3d.py)
        intrinsic = [1000.0, 1000.0, ori_w / 2.0, ori_h / 2.0]

        rgb_input, cam_model, pad, label_scale_factor = RM3D.transform_test_data_scalecano(
            rgb_origin.copy(), intrinsic, self.cfg.data_basic, PI.DEV)

        data = dict(input=rgb_input[None, :, :, :], cam_model=cam_model)
        pred_depth, _, _ = self._model.inference(data)

        # de-pad, resize back to original, rescale to metric depth (meters) — verbatim from run script
        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[pad[0]:pred_depth.shape[0] - pad[1],
                                pad[2]:pred_depth.shape[1] - pad[3]]
        pred_depth = F.interpolate(pred_depth[None, None, :, :], [ori_h, ori_w],
                                   mode="bilinear").squeeze()
        pred_depth = pred_depth * self.normalize_scale / label_scale_factor  # meters, larger = farther

        # invalid -> far (per-image max), same as run_metric3d.py, so nearness = -far = least near
        finite = torch.isfinite(pred_depth) & (pred_depth > 0)
        if finite.any():
            dmax = pred_depth[finite].max()
        else:
            dmax = torch.ones((), dtype=pred_depth.dtype, device=pred_depth.device)
        pred_depth = torch.where(finite, pred_depth, dmax)
        return pred_depth  # [ori_h, ori_w] metric depth, larger = farther

    @torch.no_grad()
    def score_relit(self, relit_bgr, xy, ohw, fold=None):
        metric_depth = self._infer_metric_depth(relit_bgr)
        near_map = -metric_depth                                  # INVERT: larger = nearer (match DAVAdapter)
        # reuse the DAV2 differentiable bilinear point sampler (input_size arg is unused inside)
        return PI.FD.sample_points(near_map, xy, ohw, None).cpu().numpy()


if __name__ == "__main__":
    # smoke test: build the model + forward a dummy 256x256 image, print class + output shapes
    ad = Metric3DAdapter("Metric3D")
    dummy_bgr = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    xy = np.array([[40, 40], [100, 80], [128, 128], [200, 150], [220, 220]], dtype=float)
    ohw = (256, 256)

    depth = ad._infer_metric_depth(dummy_bgr)
    near = ad.score_relit(dummy_bgr, xy, ohw, fold=None)

    print("model class     :", type(ad._model).__name__)
    print("depth map shape :", tuple(depth.shape), "| metric range(m):",
          float(depth.min()), "..", float(depth.max()))
    print("score_relit out :", near.shape, near.dtype, "| nearness(larger=nearer):", near)
    print("SMOKE_OK")
