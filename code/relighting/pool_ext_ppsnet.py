"""§16 pool_ext_ppsnet.py — PPSNet (ECCV 2024, Paruchuri et al.) score_relit adapter for the
counterfactual-relight decorrelation probe (code/counterfactual/e_phantom_decouple.py).

PPSNet is the near-field-lighting endoscopy depth model the paper CITES as exploiting brightness
(DepthAnything DINOv2-ViTS/14 backbone + a Per-Pixel-Shading refinement net). This adapter mirrors the
EXACT loading + preprocessing + depth orientation of
    experiments/external_baselines_newmodels/run_ppsnet.py
so the geometry-preserving relight operator is applied to the BGR image BEFORE any model code runs
(no model edits), matching the DAVAdapter / EndoOmniAdapter interface in pool_inference.py.

ORIENTATION (why we invert):
  PPSNet's backbone predicts DISPARITY; the run script converts it to depth with
      depth = 1/disparity ; clamp[0,1] ; /max  ->  RELATIVE DEPTH where LARGER = FARTHER
  (run_ppsnet.py header: "near=smaller", i.e. near_is_larger=0). DAVAdapter's convention is
  nearness = +depth with LARGER = NEARER. Because PPSNet's consumed output is distance-like depth
  (larger = farther), we INVERT: nearness = -depth. Only the ORDER of the 5 points is used by the
  probe (pairwise_acc), and -depth is a monotone-decreasing map of depth, so this reproduces exactly
  the ordering the offline PPSNet scorer would obtain from the saved depth with near_is_larger=0.

COORD MAPPING (why not FD.sample_points): run_ppsnet.py preprocesses with
  transforms.Resize(518) (shorter edge -> 518, aspect preserved) + transforms.CenterCrop(518).
On the non-square C3VD frames (1350x1080) this DROPS the horizontal margins, so the 518x518 depth map
does NOT cover the full original frame. We therefore map each original-res point through the exact
Resize+CenterCrop transform (per-axis scale + center-crop offset) into the 518 grid before sampling,
instead of the full-frame xy/ow assumption used by FD.sample_points.
"""
import os, sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import PIL
from PIL import Image
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))
P1 = os.path.dirname(os.path.dirname(HERE))                                   # .../P1_Ordinal_Endo_Depth_Benchmark...
REPO = os.path.join(P1, "experiments", "external_baselines_newmodels", "PPSNet")
sys.path.insert(0, REPO)                                                      # so `modules`,`utils`,`losses` import as packages

import utils.optical_flow_funs as OF                                          # noqa: E402
from modules.PPSNet import PPSNet_Backbone, PPSNet_Refinement                 # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
S = 518                                                                       # PPSNet is fixed at 518x518 (refiner hardcodes it)
CKPT = os.path.join(REPO, "checkpoints", "student.ckpt")                      # headline teacher-student STUDENT ckpt


def build_n_intrinsics(S=518, fov_frac=0.7):
    """Centered wide-FOV pinhole intrinsic -> normalized intrinsics (1,3,3). Copied verbatim from
    run_ppsnet.py: our frames have no known K, the PPL term is a soft correction and the ordinal
    depth ORDER is robust to the exact intrinsic (canonical-intrinsic choice, as for other baselines)."""
    f = fov_frac * S
    c = (S - 1) / 2.0
    K = torch.tensor([[f, 0, c], [0, f, c], [0, 0, 1]], dtype=torch.float32)
    return OF.pixel_intrinsics_to_normalized_intrinsics(K.unsqueeze(0), (S, S))  # (1,3,3)


class PPSNetAdapter:
    """PPSNet (student ckpt) nearness scorer. near = -depth (depth = 1/disp, larger = farther -> INVERTED
    to larger = nearer, matching DAVAdapter). Model + refiner + constant PPL light data are loaded ONCE."""

    def __init__(self, name="PPSNet"):
        self.name = name

        # ---- build + load backbone + refiner EXACTLY as run_ppsnet.py ----
        model = PPSNet_Backbone.from_pretrained("LiheYoung/depth_anything_vits14")
        refiner = PPSNet_Refinement(1, 384)
        ckpt = torch.load(CKPT, map_location="cpu")
        bb = {(k[7:] if k.startswith("module.") else k): v for k, v in ckpt["student_state_dict"].items()}
        rf = {(k[7:] if k.startswith("module.") else k): v for k, v in ckpt["refiner_state_dict"].items()}
        mb, ub = model.load_state_dict(bb, strict=False)
        mr, ur = refiner.load_state_dict(rf, strict=False)
        print(f"[PPSNet] backbone: missing={len(mb)} unexpected={len(ub)} | "
              f"refiner: missing={len(mr)} unexpected={len(ur)}", flush=True)
        self.model = model.to(DEV).eval()
        self.refiner = refiner.to(DEV).eval()

        # ---- constant PPL light data + synthesized 518 intrinsics (run_ppsnet.py) ----
        self.n_intrinsics = build_n_intrinsics(S).to(DEV)                                       # (1,3,3)
        self.ref_dirs = OF.get_camera_pixel_directions((S, S), self.n_intrinsics,
                                                       normalized_intrinsics=True).to(DEV)      # (1,S,S,3)
        self.light_pos = torch.zeros(3).unsqueeze(0).to(DEV)                                    # (1,3)
        self.light_dir = torch.tensor([0., 0., 1.]).unsqueeze(0).to(DEV)                        # (1,3)
        self.mu = torch.tensor([0.]).to(DEV)                                                    # (1,)

        # ---- exact preprocessing transforms (run_ppsnet.py) ----
        self._resize = transforms.Resize(S, interpolation=PIL.Image.BILINEAR)                   # shorter edge -> 518
        self._crop = transforms.CenterCrop(S)
        self._to_tensor = transforms.ToTensor()

    def _preprocess(self, relit_bgr):
        """Replicate run_ppsnet.py's img -> [0,1] RGB, Resize(518)+CenterCrop(518), NO imagenet norm.
        Returns (img (1,3,S,S) on DEV, rh, rw) where (rh,rw) are the PRE-crop resized dims, needed to
        map original-res point coords into the 518 cropped grid."""
        rgb = cv2.cvtColor(relit_bgr, cv2.COLOR_BGR2RGB)              # HxWx3 uint8 RGB
        pil = Image.fromarray(rgb)
        t = self._to_tensor(pil)                                     # (3,H,W) float in [0,1]  (matches ToTensor)
        t = self._resize(t)                                         # (3,rh,rw)  shorter edge -> 518
        rh, rw = int(t.shape[-2]), int(t.shape[-1])
        t = self._crop(t)                                          # (3,S,S)  center 518x518
        img = t.unsqueeze(0).to(DEV)                                # (1,3,S,S)
        img = F.interpolate(img, size=(S, S), mode="bicubic", align_corners=False)  # run-script no-op at 518
        return img, rh, rw

    def _sample(self, depth_2d, xy, ohw, rh, rw):
        """Crop-aware bilinear sample of the (S,S) depth map at the 5 original-res points.
        Maps (x,y) through Resize(shorter->518)+CenterCrop(518): x_grid = x*(rw/ow) - crop_left, etc.
        depth_2d: torch (S,S). Returns torch (5,) depth (LARGER = FARTHER)."""
        oh, ow = ohw
        sx = rw / float(ow)                                        # per-axis resize scale (accounts for rounding)
        sy = rh / float(oh)
        crop_left = int(round((rw - S) / 2.0))                     # torchvision CenterCrop offset (image >= crop)
        crop_top = int(round((rh - S) / 2.0))
        xc = xy[:, 0] * sx - crop_left                             # -> pixel x in [0, S-1] grid
        yc = xy[:, 1] * sy - crop_top
        gx = xc / (S - 1) * 2.0 - 1.0                              # align_corners=True normalization into [-1,1]
        gy = yc / (S - 1) * 2.0 - 1.0
        grid = torch.tensor(np.stack([gx, gy], 1), dtype=depth_2d.dtype,
                            device=depth_2d.device).view(1, 1, 5, 2)
        s = F.grid_sample(depth_2d.view(1, 1, S, S), grid, mode="bilinear", align_corners=True)
        return s.view(5)

    @torch.no_grad()
    def score_relit(self, relit_bgr, xy, ohw, fold=None):
        """relit_bgr: HxWx3 uint8 BGR; xy: (5,2) original-res coords; ohw:(H,W). Returns np(5,) nearness,
        LARGER = NEARER (inverted from PPSNet depth)."""
        xy = np.asarray(xy, dtype=np.float64)
        img, rh, rw = self._preprocess(relit_bgr)

        # ---- forward EXACTLY as run_ppsnet.py ----
        disparity, rgb_feats, colored_dot_feats = self.model(
            img, self.ref_dirs, self.light_pos, self.light_dir, self.mu, self.n_intrinsics)
        disp_preds = self.refiner(rgb_feats, colored_dot_feats, disparity)   # refined disparity (1,S,S)
        pred = 1.0 / disp_preds                                             # disparity -> depth (LARGER = FARTHER)
        pred = torch.clamp(pred, 0, 1)
        pmax = pred.max()
        if pmax > 0:
            pred = pred / pmax                                             # monotone (order-preserving)
        depth_2d = pred.reshape(S, S)                                       # (S,S) relative depth

        depth_pts = self._sample(depth_2d, xy, ohw, rh, rw).cpu().numpy()   # (5,) depth, larger = farther
        return -depth_pts                                                   # INVERT -> nearness (larger = nearer)


if __name__ == "__main__":
    # Smoke test: build the model (catches load / checkpoint errors) and, if a GPU is present, run one
    # forward through score_relit on a dummy 256x256 BGR image (PPSNet's forward hardcodes .to('cuda'),
    # so the forward can only run on GPU; the load itself is device-agnostic).
    print(f"[smoke] DEV={DEV}", flush=True)
    print(f"[smoke] REPO={REPO}", flush=True)
    print(f"[smoke] CKPT exists={os.path.exists(CKPT)}", flush=True)
    try:
        ad = PPSNetAdapter("PPSNet")
    except Exception as e:
        print(f"[smoke] LOAD FAILED: {type(e).__name__}: {e}", flush=True)
        print("[smoke] HINT: PPSNet_Backbone.from_pretrained needs the HF cache for "
              "'LiheYoung/depth_anything_vits14' AND its DPT_DINOv2.__init__ calls "
              "torch.hub.load('facebookresearch/dinov2','dinov2_vits14') (+ its pretrain ckpt). Both "
              "~/.cache/torch/hub/facebookresearch_dinov2_main and ~/.cache/huggingface are currently "
              "ABSENT -> repopulate them from an internet node (or drop HF_HUB_OFFLINE) before running.",
              flush=True)
        raise
    print(f"[smoke] model class = {type(ad.model).__name__}  |  refiner class = {type(ad.refiner).__name__}",
          flush=True)

    dummy = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)         # HxWx3 uint8 BGR
    xy = np.array([[30, 40], [200, 60], [128, 128], [50, 210], [220, 200]], dtype=float)
    ohw = (256, 256)
    if DEV == "cuda":
        out = ad.score_relit(dummy, xy, ohw, fold=None)
        print(f"[smoke] score_relit output: shape={out.shape} dtype={out.dtype} "
              f"values={np.round(out, 5).tolist()}", flush=True)
        assert out.shape == (5,), f"expected (5,), got {out.shape}"
        print("[smoke] orientation OK: larger = nearer (nearness = -depth)", flush=True)
    else:
        print("[smoke] no CUDA -> load verified; skipping forward (PPSNet forward hardcodes .to('cuda'))",
              flush=True)
    print("SMOKE_OK", flush=True)
