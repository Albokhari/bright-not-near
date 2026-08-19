"""Exp A controls: two reference "models" that anchor the S_relight vs dE relationship.

  oracle   perfect model. predict(..., gt=<GT depth>) returns the GT depth itself ->
           ss-aligns to a=1,b=0 -> ~0 error, and (because it ignores the image) 0 S_relight.
           Anchors the (S_relight, dE) plot at the origin: no reliance, no error change.
  lumonly  POSITIVE control. A tiny grayscale-input conv encoder-decoder that predicts a
           single-channel relative-depth map from BRIGHTNESS ALONE. Trained on C3VD-train
           where bright~=near, so it is *designed* to track illumination: under the decorr
           family (brightness equalised across depth) its prediction changes a lot
           (high S_relight) and its error rises (high dE). It is the "relies on light" point.

Both expose the SAME .predict interface as expA_dense_inference.DensePredictor, extended with an
optional `gt` kwarg so expA_run can call:
    cp.predict("oracle",  bgr, gt=gt)     # -> returns gt (HxW float, native size)
    cp.predict("lumonly", bgr)            # -> HxW float relative depth (native size)
    cp.predict("dav2",    bgr)            # -> delegated to a DensePredictor (lazy-imported)

Design notes:
  * ControlPredictor can serve as the ONE predictor for the whole run: control names are handled
    here; every other name is delegated to a lazily-constructed expA_dense_inference.DensePredictor
    (or one you pass in). This keeps expA_run's dispatch trivial.
  * lumonly's sign/scale is irrelevant downstream: expA_metrics.ss_fit allows a negative slope, so
    the network may output either "larger=nearer" or "larger=farther"; the frozen ss-fit absorbs it.
  * Model + checkpoint are loaded ONCE (lazy) to cuda if available.

Env: module PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1 + SciPy-bundle. DDPM models EXCLUDED.
"""
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
P1 = "/well/rittscher/users/fxh757/Code/Papers/P1_Ordinal_Endo_Depth_Benchmark__EndoLINA_MICCAI2026"
LUMONLY_CKPT = os.path.join(P1, "results/finetune/lumonly/lumonly.pt")

INPUT_SIZE = 256            # lumonly forward resolution (output is upsampled back to native HxW)
GRAY_MEAN = 0.5            # fixed (NOT per-image) standardisation -> keeps lumonly brightness-sensitive
GRAY_STD = 0.5
CONTROL_MODELS = ("oracle", "lumonly")


def is_control(model_name):
    """True if `model_name` is handled directly by ControlPredictor (vs delegated to DensePredictor)."""
    return model_name in CONTROL_MODELS


# --------------------------------------------------------------------------------------
# Tiny grayscale -> single-channel relative-depth CNN (the lumonly architecture).
# ~5 conv stages, encoder-decoder, fully defined here so train_lumonly.py reuses this exact class.
# --------------------------------------------------------------------------------------
def _cbr(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class TinyLumNet(nn.Module):
    """Small brightness->relative-depth network. Input (B,1,S,S) standardised grayscale; output (B,1,S,S).

    Encoder downsamples x4 (S -> S/4), decoder upsamples back to S. Deliberately tiny (~0.2M params):
    it has to lean on global brightness structure, which is exactly the reliance we want to expose."""

    def __init__(self, base=32):
        super().__init__()
        self.enc1 = _cbr(1, base)               # S
        self.down1 = _cbr(base, base, stride=2)  # S/2
        self.enc2 = _cbr(base, base * 2)          # S/2
        self.down2 = _cbr(base * 2, base * 2, stride=2)  # S/4
        self.bott = _cbr(base * 2, base * 4)      # S/4
        self.up1 = _cbr(base * 4, base * 2)       # -> S/2 (after upsample)
        self.up2 = _cbr(base * 2, base)           # -> S   (after upsample)
        self.head = nn.Conv2d(base, 1, 3, padding=1)  # linear single-channel relative depth

    def forward(self, x):
        x = self.enc1(x)
        x = self.down1(x)
        x = self.enc2(x)
        x = self.down2(x)
        x = self.bott(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2(x)
        return self.head(x)


def gray_from_bgr(bgr_uint8):
    """BGR uint8 HxWx3 -> grayscale float32 HxW in [0,1] (no per-image brightness normalisation)."""
    g = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2GRAY)
    return g.astype(np.float32) / 255.0


def preprocess_gray(bgr_uint8, size=INPUT_SIZE, device="cpu"):
    """BGR uint8 -> standardised (1,1,size,size) float tensor on `device`, matching training preprocessing."""
    g = gray_from_bgr(bgr_uint8)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_LINEAR)
    g = (g - GRAY_MEAN) / GRAY_STD
    return torch.from_numpy(g)[None, None].float().to(device)


# --------------------------------------------------------------------------------------
# ControlPredictor: oracle + lumonly, with delegation for the 7 real models.
# --------------------------------------------------------------------------------------
class ControlPredictor:
    """Adds 'oracle' and 'lumonly' to the DensePredictor.predict(model_name, bgr, ...) interface.

    predict(model_name, bgr_uint8, gt=None) -> pred_float HxW at the frame's native size.
      oracle : returns `gt` (HxW float, larger=farther). `gt` MUST be provided.
      lumonly: runs the trained TinyLumNet on the grayscale of `bgr`.
      other  : delegated to a DensePredictor (passed in, or lazily built from expA_dense_inference).
    """

    def __init__(self, dense=None, lumonly_ckpt=LUMONLY_CKPT, device=None, allow_random_lumonly=False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lumonly_ckpt = lumonly_ckpt
        self.allow_random_lumonly = allow_random_lumonly
        self._dense_obj = dense          # optional pre-built DensePredictor
        self._lum = None                 # lazy TinyLumNet
        self._lum_size = INPUT_SIZE

    # ---- oracle ----
    @staticmethod
    def _oracle(bgr_uint8, gt):
        if gt is None:
            raise ValueError("predict('oracle', bgr, gt=<GT depth HxW>) requires the gt= kwarg "
                             "(the oracle returns the ground-truth depth itself).")
        pred = np.asarray(gt, dtype=np.float32)
        pred = np.where(np.isfinite(pred), pred, 0.0).astype(np.float32)  # sanitise NaN/inf
        return pred

    # ---- lumonly ----
    def _load_lumonly(self):
        if self._lum is not None:
            return
        net = TinyLumNet()
        if os.path.exists(self.lumonly_ckpt):
            ck = torch.load(self.lumonly_ckpt, map_location="cpu")
            state = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
            net.load_state_dict(state)
            if isinstance(ck, dict):
                self._lum_size = int(ck.get("input_size", INPUT_SIZE))
        elif self.allow_random_lumonly:
            print(f"[ControlPredictor] WARNING: no checkpoint at {self.lumonly_ckpt}; "
                  f"using RANDOM-INIT lumonly (smoke only).", flush=True)
        else:
            raise FileNotFoundError(
                f"lumonly checkpoint not found: {self.lumonly_ckpt}\n"
                f"Train it first:  python3 {os.path.join(HERE, 'train_lumonly.py')}")
        self._lum = net.to(self.device).eval()

    @torch.no_grad()
    def _lumonly(self, bgr_uint8):
        self._load_lumonly()
        H, W = bgr_uint8.shape[:2]
        x = preprocess_gray(bgr_uint8, size=self._lum_size, device=self.device)
        y = self._lum(x)                                   # (1,1,s,s)
        y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
        return y[0, 0].float().cpu().numpy().astype(np.float32)

    # ---- delegation to the dense (real-model) predictor ----
    def _dense(self):
        if self._dense_obj is None:
            import expA_dense_inference as EDI      # lazy: peer module may not exist at import time
            self._dense_obj = EDI.DensePredictor()
        return self._dense_obj

    # ---- unified interface ----
    def predict(self, model_name, bgr_uint8, gt=None):
        if model_name == "oracle":
            return self._oracle(bgr_uint8, gt)
        if model_name == "lumonly":
            return self._lumonly(bgr_uint8)
        return self._dense().predict(model_name, bgr_uint8)


if __name__ == "__main__":
    # CPU smoke: exercises both controls WITHOUT a GPU or a trained checkpoint.
    print(f"[expA_controls] device probe: cuda={torch.cuda.is_available()}", flush=True)
    rng = np.random.default_rng(0)
    bgr = (rng.random((120, 160, 3)) * 255).astype(np.uint8)
    gt = (rng.random((120, 160)).astype(np.float32) * 50 + 10)

    cp = ControlPredictor(allow_random_lumonly=True, device="cpu")

    po = cp.predict("oracle", bgr, gt=gt)
    assert po.shape == gt.shape and np.allclose(po, gt), "oracle must return the GT unchanged"
    # oracle ignores the image -> identical prediction on a relit copy -> S_relight == 0
    po2 = cp.predict("oracle", (bgr.astype(np.int32) + 30).clip(0, 255).astype(np.uint8), gt=gt)
    assert np.allclose(po, po2), "oracle must be image-invariant (=> 0 S_relight)"
    print(f"oracle OK: shape {po.shape}, ==gt {np.allclose(po, gt)}, image-invariant {np.allclose(po, po2)}",
          flush=True)

    pl = cp.predict("lumonly", bgr)
    assert pl.shape == (120, 160), f"lumonly native-size output expected, got {pl.shape}"
    print(f"lumonly OK: shape {pl.shape}, dtype {pl.dtype}, finite {np.isfinite(pl).all()}", flush=True)
    print("SMOKE_OK", flush=True)
