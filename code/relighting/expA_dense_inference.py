"""Exp A :: expA_dense_inference.py — DensePredictor: one .predict(model_name, bgr) -> DENSE depth/disparity
map at the frame's NATIVE size, for every depth model in the counterfactual-relight benchmark.

WHAT THIS IS FOR
----------------
Exp A relights each C3VD/SimCol frame at FIXED geometry and measures (a) the change in dense metric-depth
error and (b) the change in the model's prediction. Both need the model's DENSE map (not the 5-point ordinal
score the ordinal probe uses). This module is the single dense-inference entry point: a frozen, lazily-cached
model zoo whose `.predict(model_name, bgr_uint8)` returns a float32 HxW array co-registered with the input
frame, so downstream ss-alignment (expA_metrics.ss_fit, scale+shift, sign-agnostic) + metrics + S_relight
can operate uniformly across models.

MODELS (7):  dav2, endoomni, dac, unidepth, metric3d, zoedepth, ppsnet

NO SIGN INVERSION HERE. We return each model's RAW native output and let ss_fit absorb scale AND sign:
  - dav2 / dac / endoomni : relative DISPARITY (larger = NEARER)   -> negative slope vs depth GT (ss_fit allows a<0)
  - unidepth / metric3d / zoedepth : metric DEPTH in meters (larger = FARTHER) -> positive slope
  - ppsnet : relative DEPTH = normalized 1/disp (larger = FARTHER) -> positive slope
S_relight (expA_run) is z-score based, hence invariant to the overall sign, so this convention is safe.

CODE REUSE (no edits to the reused modules):
  - dav2 / dac / endoomni : reuse code/c3vd_eval/save_c3vd_cropped.load_adapter(model, head) to BUILD the
    model + get its (forward_fn, mean, std), and save_c3vd_cropped.preprocess (Resize(shorter->518) +
    CenterCrop(518)) for the exact paper FOV. The 518x518 crop map is pasted back into the native canvas
    (see FOV / PASTE-BACK below).
  - unidepth / metric3d / zoedepth / ppsnet : reuse the already-validated adapters
    code/counterfactual/pool_ext_{unidepth,metric3d,zoedepth,ppsnet}.py. Their score_relit() computes a dense
    map then samples 5 points + negates; here we call the SAME loaded model/forward but return the dense map
    BEFORE the 5-point sampling and BEFORE negation (via the adapter's own internal method / attributes).

FOV / PASTE-BACK (crop models: dav2, dac, endoomni, ppsnet):
  These 4 preprocess with Resize(shorter edge -> 518) + CenterCrop(518). That crop is exactly the center
  min(H,W)-square of the native frame (shorter edge -> 518 then a 518 square == the center square of side
  min(H,W) in native pixels). We therefore resize the crop's dense map to (min(H,W), min(H,W)) and paste it
  into the center square of an all-NaN native canvas. On C3VD (1350x1080) the left/right margins (cols
  [0,135) and [1215,1350)) are outside the model FOV and are returned as NaN. On SimCol (475x475, square) the
  crop covers the whole frame -> no NaN.
  IMPORTANT for the caller: predict() may return NaN outside a crop model's FOV. expA_metrics.ss_fit /
  metrics MUST restrict to (mask & np.isfinite(pred)); the relight ΔE comparison is only defined on the
  finite intersection. The 3 full-frame models (unidepth/metric3d/zoedepth) return a fully-finite native map.

ENV (set at import; matches the project GPU jobs):
  HF_HOME=/well/rittscher/users/fxh757/system/temp/cache/huggingface, HF_HUB_OFFLINE=1, XFORMERS_DISABLED=1,
  and /well/rittscher/users/fxh757/system/decouple_pylibs on sys.path (unidepth/metric3d/zoedepth deps).

GPU: every real forward needs CUDA (checkpoints load to cuda; PPSNet's forward hardcodes .to('cuda')). This
module py_compiles + imports on CPU (all model imports are lazy), but predict() for any model needs a GPU.
"""
import os
import sys

# ---- env (before any torch / HF import) -------------------------------------------------------------
os.environ.setdefault("HF_HOME", "/well/rittscher/users/fxh757/system/temp/cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("XFORMERS_DISABLED", "1")   # DINOv2 plain-attention fallback (matches the job env)
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))              # .../code/counterfactual
CODE_DIR = os.path.dirname(HERE)                               # .../code
C3VD_EVAL = os.path.join(CODE_DIR, "c3vd_eval")                # save_c3vd_cropped lives here
DECOUPLE = "/well/rittscher/users/fxh757/system/decouple_pylibs"

for _p in (HERE, C3VD_EVAL, DECOUPLE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# cv2 is only needed inside predict(); import lazily so a bare `import expA_dense_inference` stays light.
try:
    import cv2  # noqa: F401
    _HAVE_CV2 = True
except Exception:                                             # pragma: no cover
    _HAVE_CV2 = False

DEV = "cuda" if torch.cuda.is_available() else "cpu"

SUPPORTED = ("dav2", "endoomni", "dac", "unidepth", "metric3d", "zoedepth", "ppsnet")
CROP_MODELS = ("dav2", "endoomni", "dac", "ppsnet")           # Resize(518)+CenterCrop(518) -> center-square FOV
FULL_MODELS = ("unidepth", "metric3d", "zoedepth")           # native full-frame coverage


def _cv2():
    import cv2
    return cv2


class DensePredictor:
    """Lazily-cached dense-inference zoo for the Exp A counterfactual-relight benchmark.

    predict(model_name, bgr_uint8, gt=None) -> float32 HxW dense map at the input frame's NATIVE size.
      * model_name in SUPPORTED (case-insensitive).
      * bgr_uint8 : HxWx3 uint8 BGR (relighting is already applied to the pixels before calling predict).
      * gt        : ignored by the 7 real models (present so expA_controls.py can add an 'oracle' predictor
                    with the SAME .predict signature).
      * returns   : the model's RAW native depth/disparity (NO sign flip — ss_fit absorbs scale+sign). Crop
                    models return NaN outside their center-square FOV.

    Each model is built ONCE on first use and cached (keyed by model_name). For the DINOv2 relative models
    (dav2/dac/endoomni) the fine-tune head is selectable via head_map (default 'base' = pretrained head).
    """

    def __init__(self, head_map=None, device=None):
        # head_map: {model_name -> head label understood by save_c3vd_cropped.load_adapter}, e.g.
        #   {"dav2": "dav2_fold0", "dac": "dac_fold0", "endoomni": "base"}. Default "base" for all.
        self.head_map = dict(head_map or {})
        self.device = device or DEV
        self._cache = {}

    # ---- public API ---------------------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, model_name, bgr, gt=None):
        name = str(model_name).lower()
        if name not in SUPPORTED:
            raise ValueError(f"unknown model_name {model_name!r}; supported = {SUPPORTED}")
        if not _HAVE_CV2:
            raise RuntimeError("cv2 (opencv) is required for predict() but failed to import")
        bgr = np.asarray(bgr)
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(f"bgr must be HxWx3 uint8, got shape {bgr.shape}")
        H, W = int(bgr.shape[0]), int(bgr.shape[1])
        if name in ("dav2", "dac", "endoomni"):
            out = self._predict_dinov2(name, bgr, H, W)
        elif name == "ppsnet":
            out = self._predict_ppsnet(bgr, H, W)
        elif name == "unidepth":
            out = self._predict_unidepth(bgr, H, W)
        elif name == "metric3d":
            out = self._predict_metric3d(bgr, H, W)
        elif name == "zoedepth":
            out = self._predict_zoedepth(bgr, H, W)
        else:                                                 # pragma: no cover (guarded above)
            raise ValueError(name)
        out = np.asarray(out, dtype=np.float32)
        assert out.shape == (H, W), f"{name}: predict returned {out.shape}, expected {(H, W)}"
        return out

    def supported(self):
        return tuple(SUPPORTED)

    # ---- DINOv2 relative models (dav2 / dac / endoomni) via save_c3vd_cropped ------------------------
    def _predict_dinov2(self, name, bgr, H, W):
        cv2 = _cv2()
        from PIL import Image
        key = ("dinov2", name)
        if key not in self._cache:
            import save_c3vd_cropped as SC                    # lazy: pulls finetune_* + the model ckpt
            head = self.head_map.get(name, "base")
            fwd, mean, std = SC.load_adapter(name, head)      # builds model ONCE on SC.DEV, returns forward+norm
            self._cache[key] = (SC, fwd, mean, std, head)
        SC, fwd, mean, std, head = self._cache[key]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = SC.preprocess(Image.fromarray(rgb), mean, std)    # (1,3,518,518) Resize(518)+CenterCrop(518) on SC.DEV
        crop = fwd(x).float().detach().cpu().numpy().astype(np.float32)   # [h',w'] disparity over the 518 crop
        crop = self._fill_nonfinite(crop)
        return self._paste_center_square(crop, H, W)          # native canvas, NaN outside the center square

    # ---- PPSNet (near-field-shading endoscopy model) via pool_ext_ppsnet -----------------------------
    def _predict_ppsnet(self, bgr, H, W):
        key = ("ppsnet",)
        if key not in self._cache:
            import pool_ext_ppsnet as PP
            self._cache[key] = (PP, PP.PPSNetAdapter("PPSNet"))
        PP, ad = self._cache[key]
        # replicate run_ppsnet.py's forward EXACTLY (== pool_ext_ppsnet.score_relit) up to the dense depth,
        # but WITHOUT the 5-point sample and WITHOUT the nearness negation.
        img, _rh, _rw = ad._preprocess(bgr)                   # (1,3,S,S) on DEV
        disparity, rgb_feats, colored_dot_feats = ad.model(
            img, ad.ref_dirs, ad.light_pos, ad.light_dir, ad.mu, ad.n_intrinsics)
        disp_preds = ad.refiner(rgb_feats, colored_dot_feats, disparity)  # refined disparity
        pred = 1.0 / disp_preds
        pred = torch.clamp(pred, 0, 1)                        # depth (larger = FARTHER), relative
        pmax = pred.max()
        if pmax > 0:
            pred = pred / pmax                                # monotone normalize (order-preserving)
        crop = pred.reshape(PP.S, PP.S).detach().cpu().numpy().astype(np.float32)
        crop = self._fill_nonfinite(crop)
        return self._paste_center_square(crop, H, W)

    # ---- UniDepth-V2 (metric) via pool_ext_unidepth --------------------------------------------------
    def _predict_unidepth(self, bgr, H, W):
        cv2 = _cv2()
        key = ("unidepth",)
        if key not in self._cache:
            import pool_ext_unidepth as UD
            self._cache[key] = UD.UniDepthAdapter("UniDepth")
        ad = self._cache[key]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)            # HxWx3 uint8 RGB (infer() std-izes internally)
        rgb_t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)   # uint8 CHW
        camera = ad._camera(H, W)                             # canonical pinhole (relative order only)
        pred = ad._model.infer(rgb_t, camera)
        depth = pred["depth"].squeeze().float().detach().cpu().numpy().astype(np.float32)  # meters, larger=farther
        depth = self._fill_nonfinite(depth)
        return self._to_native(depth, H, W)

    # ---- Metric3D (metric) via pool_ext_metric3d -----------------------------------------------------
    def _predict_metric3d(self, bgr, H, W):
        key = ("metric3d",)
        if key not in self._cache:
            import pool_ext_metric3d as M3
            self._cache[key] = M3.Metric3DAdapter("Metric3D")
        ad = self._cache[key]
        depth = ad._infer_metric_depth(bgr)                  # [H,W] torch, native res, invalid->far already applied
        depth = depth.detach().cpu().numpy().astype(np.float32)
        depth = self._fill_nonfinite(depth)
        return self._to_native(depth, H, W)

    # ---- ZoeDepth (metric) via pool_ext_zoedepth -----------------------------------------------------
    def _predict_zoedepth(self, bgr, H, W):
        cv2 = _cv2()
        from PIL import Image
        key = ("zoedepth",)
        if key not in self._cache:
            import pool_ext_zoedepth as ZD
            ad = ZD.ZoeDepthAdapter("ZoeDepth")
            ad._get()                                        # build the heavy ViT-L once
            self._cache[key] = ad
        ad = self._cache[key]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = ad._get().infer_pil(Image.fromarray(rgb))    # HxW metric depth at native input res
        depth = np.asarray(depth, dtype=np.float32)
        depth = self._fill_nonfinite(depth)
        return self._to_native(depth, H, W)

    # ---- shared helpers -----------------------------------------------------------------------------
    @staticmethod
    def _fill_nonfinite(a):
        """Replace non-finite pixels with the finite max (== farthest for a depth map), matching the
        invalid->far rule the pool_ext adapters use. Operates on the model's raw (pre-paste) map."""
        a = np.asarray(a, dtype=np.float32)
        finite = np.isfinite(a)
        if finite.all():
            return a
        fill = float(a[finite].max()) if finite.any() else 0.0
        return np.where(finite, a, fill).astype(np.float32)

    @staticmethod
    def _to_native(a, H, W):
        """Resize a full-frame dense map to the native (H,W) if the model emitted a different size."""
        a = np.asarray(a, dtype=np.float32)
        if a.shape == (H, W):
            return a
        t = torch.from_numpy(np.ascontiguousarray(a)).float().view(1, 1, a.shape[0], a.shape[1])
        r = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
        return r.view(H, W).numpy().astype(np.float32)

    @staticmethod
    def _paste_center_square(crop, H, W):
        """Paste a crop model's 518-square dense map into the NATIVE canvas. Resize(shorter->518)+
        CenterCrop(518) == the center min(H,W)-square of the frame, so we resize the crop map to
        (S,S), S=min(H,W), and place it in the center; pixels outside (the dropped margins) are NaN."""
        crop = np.asarray(crop, dtype=np.float32)
        S = int(min(H, W))
        t = torch.from_numpy(np.ascontiguousarray(crop)).float().view(1, 1, crop.shape[0], crop.shape[1])
        r = F.interpolate(t, size=(S, S), mode="bilinear", align_corners=False).view(S, S).numpy().astype(np.float32)
        out = np.full((H, W), np.nan, dtype=np.float32)
        top = (H - S) // 2
        left = (W - S) // 2
        out[top:top + S, left:left + S] = r
        return out


# =====================================================================================================
# Smoke test / self-test.
#   python3 expA_dense_inference.py --selftest         # CPU-only: geometry/helper checks (no models)
#   python3 expA_dense_inference.py --model dav2        # GPU: build + one forward on a real/dummy frame
# =====================================================================================================
def _selftest():
    """CPU-only validation of the non-GPU logic (paste-back geometry, native resize, non-finite fill)."""
    dp = DensePredictor()
    # 1) center-square paste on a C3VD-shaped landscape frame: NaN margins, finite center square.
    crop = np.arange(518 * 518, dtype=np.float32).reshape(518, 518)
    out = dp._paste_center_square(crop, 1080, 1350)
    assert out.shape == (1080, 1350)
    assert np.isnan(out[:, :135]).all() and np.isnan(out[:, 1215:]).all(), "margins must be NaN"
    assert np.isfinite(out[:, 135:1215]).all(), "center square must be finite"
    assert np.isnan(out).sum() == 1080 * (1350 - 1080), "exactly the two side margins are NaN"
    # 2) square frame (SimCol): no NaN.
    outsq = dp._paste_center_square(crop, 475, 475)
    assert outsq.shape == (475, 475) and np.isfinite(outsq).all(), "square frame must be fully finite"
    # 3) full-frame resize is a no-op at native size and reshapes otherwise.
    a = np.random.rand(475, 475).astype(np.float32)
    assert dp._to_native(a, 475, 475) is a or np.allclose(dp._to_native(a, 475, 475), a)
    b = np.random.rand(392, 518).astype(np.float32)
    assert dp._to_native(b, 1080, 1350).shape == (1080, 1350)
    # 4) non-finite fill -> finite max.
    z = np.array([[1.0, np.nan], [np.inf, 3.0]], dtype=np.float32)
    f = dp._fill_nonfinite(z)
    assert np.isfinite(f).all() and f.max() == 3.0 and f[0, 1] == 3.0
    print("SELFTEST_OK (paste-back / native-resize / nonfinite-fill all pass on CPU)")


def _smoke_model(name):
    import argparse  # noqa
    dp = DensePredictor()
    print(f"[smoke] model={name} DEV={DEV}", flush=True)
    if DEV != "cuda":
        print("[smoke] no CUDA on this node -> model build + forward need a GPU; import/compile only here.",
              flush=True)
        return
    # prefer a real C3VD frame if present, else a dummy 1350x1080 BGR frame.
    cv2 = _cv2()
    import glob
    c3 = sorted(glob.glob("/well/rittscher/users/fxh757/Datasets/PUBLIC_ENDO/Depth_split_2/test/images/*.png"))
    if c3:
        bgr = cv2.imread(c3[0])
    else:
        bgr = (np.random.rand(1080, 1350, 3) * 255).astype(np.uint8)
    pred = dp.predict(name, bgr)
    finite = np.isfinite(pred)
    vals = pred[finite]
    print(f"[smoke] pred shape={pred.shape} dtype={pred.dtype} "
          f"finite_frac={finite.mean():.3f} range=[{vals.min():.4g},{vals.max():.4g}]", flush=True)
    assert pred.shape == bgr.shape[:2]
    print("SMOKE_OK", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Exp A dense-inference zoo smoke test")
    ap.add_argument("--selftest", action="store_true", help="CPU-only geometry/helper checks (no models)")
    ap.add_argument("--model", choices=list(SUPPORTED), help="build + one forward (needs a GPU)")
    args = ap.parse_args()
    if args.selftest or not args.model:
        _selftest()
    if args.model:
        _smoke_model(args.model)
