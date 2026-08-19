#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bench_depth_ranking_kvasirseg_refined.py

Refined depth-ranking benchmark (5 pixel points per image) for Kvasir-SEG test set.

What it does
------------
For each annotated image:
  1) Run diffusion sampling to generate prediction (depth, and optionally RGB if model outputs RGBD).
  2) Map the 5 annotated points from original image coords -> resized square coords (input_size x input_size).
  3) Read predicted depth values at those 5 pixels and compute a predicted ordering.
  4) Compare to GT ranks (1..5) using:
       - point_rank_accuracy (fraction of points with correct rank)
       - correct_count_out_of_5 (0..5)
       - exact_match (all 5 ranks correct)
       - pairwise_accuracy (10 ordered pairs)
       - kendall_tau
       - spearman_rho

Per model per seed we also compute a distribution:
  n_img_correct_5 ... n_img_correct_0
meaning how many images had 5/4/3/2/1/0 points correctly ranked.

Saving outputs
--------------
For each model+seed+image, saves:
  - pred_depth_u16/<stem>.png            (uint16 depth in [0,1])
  - pred_depth_vis/<stem>.png            (colored visualization)
  - pred_rgb_u8/<stem>.png               (only if model outputs RGB channels; uint8 RGB)

If reusing predictions:
  - Depth evaluation can reuse pred_depth_u16.
  - If RGB outputs are missing and you want them, set --overwrite_missing_outputs True.

Custom plot labels
------------------
--label "match_substring=Nice Label"
If no match rule hits, default label is "{in_channels}ch".
Collisions auto get "#2", "#3", ...

IMPORTANT
---------
- Keeps DEFAULT_WEIGHT_SPECS list exactly as provided.
- Assumes your diffusion_model.* modules and trainer_*ch modules are available.
"""

# =============================================================================
# PRE-PARSE --gpus EARLY (so CUDA_VISIBLE_DEVICES is set before torch import)
# =============================================================================
import os
import sys


def _preparse_flag_value(argv, flag, default=None):
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            return argv[i + 1]
    return default


_DEFAULT_GPUS = "3,4"
_gpus_arg = _preparse_flag_value(sys.argv, "--gpus", None)

if _gpus_arg is None:
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if inherited:
        _gpus_arg = inherited
    else:
        _gpus_arg = _DEFAULT_GPUS

# In multiprocessing spawn, child often inherits CUDA_VISIBLE_DEVICES; do not clobber it.
if isinstance(_gpus_arg, str) and _gpus_arg.strip():
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpus_arg.strip()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# =============================================================================
# STANDARD LIBS
# =============================================================================
import argparse
import csv
import hashlib
import importlib
import json
import math
import random
import re
import shutil
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# =============================================================================
# PLOTTING
# =============================================================================
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# TORCH
# =============================================================================
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

# =============================================================================
# DEFAULT PATHS (YOUR PROJECT)
# =============================================================================
DEFAULT_IMAGES_ROOT = "/well/rittscher/users/fxh757/Code/Kvasir_Ranking/Merged_PASS/images"
DEFAULT_EDGES_ROOT = "/well/rittscher/users/fxh757/Code/Kvasir_Ranking/Merged_PASS/edges_1.5"
DEFAULT_ANNOTATION_JSON = (
    "/well/rittscher/users/fxh757/Code/Kvasir_Ranking/Merged_PASS/merged_pass_consensus_gt.json"
)
DEFAULT_OUTPUT_ROOT = "/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/final_ranking_eval_kvasirseg_paper/rational_loss"

# =============================================================================
# MODEL QUICK CONFIG (kept compatible with your training code)
# =============================================================================
DEFAULT_INPUT_SIZE = 256
DEFAULT_ATTENTION_RESOLUTIONS = "64,32,16,8,4,2,1"
DEFAULT_UNET_NAME = "unet_512_v2"
DEFAULT_IN_CHANNELS = 8  # fallback if spec doesn't override
SEEDS = "0"

# =============================================================================
# DEFAULT WEIGHT SPECS (KEEP AS-IS)
# =============================================================================
DEFAULT_WEIGHT_SPECS = [
    # f"unet_512_v2@{DEFAULT_INPUT_SIZE}@{DEFAULT_ATTENTION_RESOLUTIONS}@2ch:/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/results_2ch_256_2nd",
    # f"unet_512_v2@{DEFAULT_INPUT_SIZE}@{DEFAULT_ATTENTION_RESOLUTIONS}@4ch:/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/results_4ch_256_2nd",
    # f"unet_512_v2@{DEFAULT_INPUT_SIZE}@{DEFAULT_ATTENTION_RESOLUTIONS}@5ch:/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/results_5ch_256_2nd",
    # f"unet_512_v2@{DEFAULT_INPUT_SIZE}@{DEFAULT_ATTENTION_RESOLUTIONS}@7ch:/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/results_7ch_256_2nd",
    # f"unet_512_v2@{DEFAULT_INPUT_SIZE}@{DEFAULT_ATTENTION_RESOLUTIONS}@8ch:/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/results_8ch_256_2nd",

    # NoPhotoAug models (for checking if they are better/worse than the ones with photo aug):
    # f"unet_512_v2@{DEFAULT_INPUT_SIZE}@{DEFAULT_ATTENTION_RESOLUTIONS}@2ch:/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/results_2ch_256_2nd_noPhotoAug",
    # f"unet_512_v2@{DEFAULT_INPUT_SIZE}@{DEFAULT_ATTENTION_RESOLUTIONS}@4ch:/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/results_4ch_256_2nd_noPhotoAug",
    # f"unet_512_v2@{DEFAULT_INPUT_SIZE}@{DEFAULT_ATTENTION_RESOLUTIONS}@5ch:/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/results_5ch_256_2nd_noPhotoAug",
    # f"unet_512_v2@{DEFAULT_INPUT_SIZE}@{DEFAULT_ATTENTION_RESOLUTIONS}@7ch:/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/results_7ch_256NoPhotoAug_3nd",
    # f"unet_512_v2@{DEFAULT_INPUT_SIZE}@{DEFAULT_ATTENTION_RESOLUTIONS}@8ch:/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/results_8ch_256_2nd_noPhotoAug", 

    # MUST CHANGE trainer_2ch.py for these to trainer_2ch_depth.py:
    # f"unet_512_v2@{DEFAULT_INPUT_SIZE}@{DEFAULT_ATTENTION_RESOLUTIONS}@2ch:/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/results_2ch_256_2nd__noPhotoAug_rational_weighted_mse000",
    # f"unet_512_v2@{DEFAULT_INPUT_SIZE}@{DEFAULT_ATTENTION_RESOLUTIONS}@2ch:/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/results_2ch_256_2nd__noPhotoAug_rational_weighted_mse040",
    # f"unet_512_v2@{DEFAULT_INPUT_SIZE}@{DEFAULT_ATTENTION_RESOLUTIONS}@2ch:/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/results_2ch_256_2nd__noPhotoAug_rational_weighted_mse050",
    # f"unet_512_v2@{DEFAULT_INPUT_SIZE}@{DEFAULT_ATTENTION_RESOLUTIONS}@2ch:/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/results_2ch_256_2nd__noPhotoAug_rational_weighted_mse060",
    # f"unet_512_v2@{DEFAULT_INPUT_SIZE}@{DEFAULT_ATTENTION_RESOLUTIONS}@2ch:/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/results_2ch_256_2nd__noPhotoAug_rational_weighted_mse070",
    # f"unet_512_v2@{DEFAULT_INPUT_SIZE}@{DEFAULT_ATTENTION_RESOLUTIONS}@2ch:/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/results_2ch_256_2nd__noPhotoAug_rational_weighted_mse080",
    # f"unet_512_v2@{DEFAULT_INPUT_SIZE}@{DEFAULT_ATTENTION_RESOLUTIONS}@2ch:/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/results_2ch_256_2nd__noPhotoAug_rational_weighted_mse090",
    # f"unet_512_v2@{DEFAULT_INPUT_SIZE}@{DEFAULT_ATTENTION_RESOLUTIONS}@2ch:/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/results_2ch_256_2nd__noPhotoAug_rational_weighted_mse095",
    f"unet_512_v2@{DEFAULT_INPUT_SIZE}@{DEFAULT_ATTENTION_RESOLUTIONS}@2ch:/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/results_2ch_256_2nd__noPhotoAug_rational_weighted_mse098",

    f"unet_512_v2@{DEFAULT_INPUT_SIZE}@{DEFAULT_ATTENTION_RESOLUTIONS}@4ch:/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/results_4ch_256_2nd_noPhotoAug_rational_weighted_mse098",
    f"unet_512_v2@{DEFAULT_INPUT_SIZE}@{DEFAULT_ATTENTION_RESOLUTIONS}@5ch:/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/results_5ch_256_2nd_noPhotoAug_rational_weighted_mse098",
    f"unet_512_v2@{DEFAULT_INPUT_SIZE}@{DEFAULT_ATTENTION_RESOLUTIONS}@7ch:/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/results_7ch_256NoPhotoAug_3nd_rational_weighted_mse098",
    f"unet_512_v2@{DEFAULT_INPUT_SIZE}@{DEFAULT_ATTENTION_RESOLUTIONS}@8ch:/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/results_8ch_256_2nd_noPhotoAug_rational_weighted_mse098",


    # MUST CHANGE trainer_2ch.py for these to trainer_2ch_depth.py:
    # f"unet_512_v2@{DEFAULT_INPUT_SIZE}@{DEFAULT_ATTENTION_RESOLUTIONS}@2ch:/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/results_2ch_256_2nd_data_weighted_mse/model-30.pt",
    # f"unet_512_v2@{DEFAULT_INPUT_SIZE}@{DEFAULT_ATTENTION_RESOLUTIONS}@2ch:/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/results_2ch_256_2nd_weighted_mse/model-50.pt",
    # f"unet_512_v2@{DEFAULT_INPUT_SIZE}@{DEFAULT_ATTENTION_RESOLUTIONS}@2ch:/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/results_2ch_256_2nd_MSE/model-50.pt",
    # f"unet_512_v2@{DEFAULT_INPUT_SIZE}@{DEFAULT_ATTENTION_RESOLUTIONS}@2ch:/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/results_2ch_256_2nd__noPhotoAug_data_weighted_mse_unfrozenNoAug",
    # f"unet_512_v2@{DEFAULT_INPUT_SIZE}@{DEFAULT_ATTENTION_RESOLUTIONS}@2ch:/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/results_2ch_256_2nd__noPhotoAug_weighted_mse_unfrozenNoAug",
    # f"unet_512_v2@{DEFAULT_INPUT_SIZE}@{DEFAULT_ATTENTION_RESOLUTIONS}@2ch:/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/results_2ch_256_2nd__noPhotoAug_mse_unfrozenNoAug",
    # f"unet_512_v2@{DEFAULT_INPUT_SIZE}@{DEFAULT_ATTENTION_RESOLUTIONS}@2ch:/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/results_2ch_256_2nd__noPhotoAug_data_weighted_mse_NoAug_paper",
    # f"unet_512_v2@{DEFAULT_INPUT_SIZE}@{DEFAULT_ATTENTION_RESOLUTIONS}@2ch:/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/results_2ch_256_2nd__noPhotoAug_weighted_mse_NoAug",
    # f"unet_512_v2@{DEFAULT_INPUT_SIZE}@{DEFAULT_ATTENTION_RESOLUTIONS}@2ch:/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/results_2ch_256_2nd__noPhotoAug_mse_NoAug",
]

# =============================================================================
# UNET MODULE MAP (your project)
# =============================================================================
UNET_NAME_TO_MODULE = {
    "unet_512_v2": "diffusion_model.unet_512_v2",
    "unet_512_v3": "diffusion_model.unet_512_v3",
    "unet_512_v4": "diffusion_model.unet_512_v4",
    "unet_512_v5": "diffusion_model.unet_512_v5",
    "unet_512_v6": "diffusion_model.unet_512_v6",
}


def get_create_model(unet_name: str):
    name = str(unet_name).strip()
    if not name:
        raise ValueError("unet_name is empty.")
    mod_path = UNET_NAME_TO_MODULE.get(name, None)
    if mod_path is None:
        known = ", ".join(sorted(UNET_NAME_TO_MODULE.keys()))
        raise ValueError(f"Unknown unet_name='{name}'. Known: {known}. Extend UNET_NAME_TO_MODULE if needed.")
    mod = importlib.import_module(mod_path)
    if not hasattr(mod, "create_model"):
        raise AttributeError(f"Module '{mod_path}' does not define create_model().")
    return getattr(mod, "create_model")


def infer_out_channels_from_in_channels(in_channels: int) -> int:
    ic = int(in_channels)
    if ic in (5, 7, 8):
        return 4
    if ic in (2, 4):
        return 1
    raise ValueError(f"Unsupported in_channels={ic}. Expected one of 2,4,5,7,8.")


def get_gaussian_diffusion_class(in_channels: int):
    ic = int(in_channels)
    if ic == 8:
        mod = importlib.import_module("diffusion_model.trainer_8ch")
    elif ic == 7:
        mod = importlib.import_module("diffusion_model.trainer_7ch_depth")
    elif ic == 5:
        mod = importlib.import_module("diffusion_model.trainer_5ch_depth")
    elif ic == 4:
        mod = importlib.import_module("diffusion_model.trainer_4ch_depth")
    elif ic == 2:
        mod = importlib.import_module("diffusion_model.trainer_2ch_depth")
    else:
        raise ValueError(f"Unsupported in_channels={ic}. Expected one of 2,4,5,7,8.")
    if not hasattr(mod, "GaussianDiffusion"):
        raise AttributeError(f"Module '{mod.__name__}' does not define GaussianDiffusion.")
    return getattr(mod, "GaussianDiffusion")


# =============================================================================
# UTILS
# =============================================================================
def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower().strip()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def apply_torch_determinism(args):
    if not bool(getattr(args, "deterministic", False)):
        return
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def set_all_seeds(seed: int):
    seed = int(seed) & 0x7FFFFFFF
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_hash32(s: str) -> int:
    b = str(s).encode("utf-8", errors="ignore")
    d = hashlib.md5(b).digest()
    return int.from_bytes(d[:4], byteorder="little", signed=False)


def per_image_seed(base_seed: int, stem: str) -> int:
    return (int(base_seed) + int(stable_hash32(stem))) & 0x7FFFFFFF


def parse_seed_list(seed_single: Optional[int], seeds_csv: str) -> List[int]:
    if seed_single is not None:
        return [int(seed_single)]
    s = (seeds_csv or "").strip()
    if not s:
        return [0]
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out if out else [0]


def slugify(s: str, max_len: int = 160) -> str:
    s = str(s)
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    slug = slug.strip("_")
    if len(slug) > max_len:
        slug = slug[:max_len]
    return slug or "item"


def normalize_attention_resolutions(attn: Optional[str]) -> Optional[str]:
    if attn is None:
        return None
    s = str(attn).strip()
    if not s:
        return None
    nums = re.findall(r"\d+", s)
    if nums:
        return ",".join(nums)
    s = s.replace("|", ",")
    s = re.sub(r"[()\[\]{}]", "", s)
    s = re.sub(r"\s+", "", s)
    s = re.sub(r",+", ",", s).strip(",")
    return s or None


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


# =============================================================================
# DATA CLASSES FOR ANNOTATIONS
# =============================================================================
@dataclass
class AnnPoint:
    point_id: int
    x: float
    y: float
    rank: int


@dataclass
class AnnImage:
    filename: str
    path: str
    width: int
    height: int
    points: List[AnnPoint]


_WARNED_ONCE: set = set()


def _warn_once(key: str, msg: str):
    if key in _WARNED_ONCE:
        return
    _WARNED_ONCE.add(key)
    print(msg)


# =============================================================================
# ANNOTATION LOADING
# =============================================================================
def load_annotation_json(annotation_path: str, images_root: str, max_images: int = 0) -> List[AnnImage]:
    if not os.path.isfile(annotation_path):
        raise FileNotFoundError(f"annotation_json not found: {annotation_path}")
    if not os.path.isdir(images_root):
        raise FileNotFoundError(f"images_root not found: {images_root}")

    with open(annotation_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "images" not in data:
        raise ValueError("Annotation JSON must be a dict with an 'images' key.")

    ann_images: List[AnnImage] = []
    for rec in data.get("images", []):
        try:
            fn = str(rec["filename"])
            w = int(rec["width"])
            h = int(rec["height"])
            pts_raw = rec["points"]
        except Exception as e:
            _warn_once(f"bad_img_rec::{id(rec)}", f"[WARN] Skipping malformed image record: {e}")
            continue

        img_path = os.path.join(images_root, fn)
        if not os.path.isfile(img_path):
            _warn_once(f"missing_img::{fn}", f"[WARN] Image file referenced by annotation not found, skipping: {img_path}")
            continue

        # Sanity check recorded size vs actual size; use actual for mapping if mismatch
        try:
            with Image.open(img_path) as im:
                aw, ah = im.size
            if aw != w or ah != h:
                _warn_once(
                    f"size_mismatch::{fn}",
                    f"[WARN] Size mismatch for {fn}: json says ({w},{h}) but file is ({aw},{ah}). Using ACTUAL size.",
                )
                w, h = aw, ah
        except Exception:
            pass

        pts: List[AnnPoint] = []
        try:
            for p in pts_raw:
                pts.append(
                    AnnPoint(
                        point_id=int(p["point_id"]),
                        x=float(p["x"]),
                        y=float(p["y"]),
                        rank=int(p["rank"]),
                    )
                )
        except Exception as e:
            _warn_once(f"bad_pts::{fn}", f"[WARN] Skipping {fn} due to malformed points: {e}")
            continue

        if len(pts) != 5:
            _warn_once(f"npts::{fn}", f"[WARN] {fn}: expected 5 points but got {len(pts)}. Keeping it anyway.")

        # Basic rank sanity
        ranks = [p.rank for p in pts]
        if any((r < 1 or r > 5) for r in ranks):
            _warn_once(f"rank_range::{fn}", f"[WARN] {fn}: some GT ranks are outside [1..5].")

        ann_images.append(AnnImage(filename=fn, path=img_path, width=w, height=h, points=pts))

        if max_images and len(ann_images) >= int(max_images):
            break

    if not ann_images:
        raise RuntimeError("No usable annotated images found (after filtering missing files).")

    return ann_images


# =============================================================================
# EDGE MATCHING
# =============================================================================
def _find_by_stem_any_ext(root: str, stem: str) -> str:
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    for e in exts:
        p = os.path.join(root, stem + e)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Could not find file for stem '{stem}' under '{root}'.")


def find_matching_edge(img_path: str, edge_root: str) -> str:
    stem = os.path.splitext(os.path.basename(img_path))[0]
    candidate = os.path.join(edge_root, os.path.basename(img_path))
    if os.path.isfile(candidate):
        return candidate
    return _find_by_stem_any_ext(edge_root, stem)


# =============================================================================
# CONDITION TENSORS (same logic as your benchmark)
# =============================================================================
def scale_tensor_01_to_m11(t: torch.Tensor) -> torch.Tensor:
    return t * 2.0 - 1.0


def _rgb_tensor_from_path(img_path: str, input_size: int) -> torch.Tensor:
    rgb = Image.open(img_path).convert("RGB")
    rgb = rgb.resize((input_size, input_size), resample=Image.BILINEAR)
    return transforms.ToTensor()(rgb)


def _gray_from_rgb_tensor(rgb_t01: torch.Tensor) -> torch.Tensor:
    if rgb_t01.ndim != 3 or rgb_t01.shape[0] != 3:
        raise ValueError(f"Expected rgb_t01 [3,H,W], got {tuple(rgb_t01.shape)}")
    r = rgb_t01[0:1]
    g = rgb_t01[1:2]
    b = rgb_t01[2:3]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.clamp(0.0, 1.0)


def build_condition_tensor(img_path: str, input_size: int, in_channels: int, args) -> torch.Tensor:
    ic = int(in_channels)
    if ic not in (2, 4, 5, 7, 8):
        raise ValueError(f"Unsupported in_channels={ic}. Expected one of 2,4,5,7,8.")

    rgb_t01 = _rgb_tensor_from_path(img_path, input_size)
    rgb_tm11 = scale_tensor_01_to_m11(rgb_t01)

    if ic == 8:
        edge_root = str(getattr(args, "edges_root", "") or "").strip()
        if not edge_root:
            raise ValueError("8ch mode requires --edges_root to be set.")
        edge_path = find_matching_edge(img_path, edge_root)
        edge = Image.open(edge_path).convert("L")
        edge = edge.resize((input_size, input_size), resample=Image.NEAREST)
        edge_t01 = transforms.ToTensor()(edge)
        edge_tm11 = scale_tensor_01_to_m11(edge_t01)
        return torch.cat([rgb_tm11, edge_tm11], dim=0)

    if ic in (7, 4):
        return rgb_tm11

    if ic in (5, 2):
        gray_t01 = _gray_from_rgb_tensor(rgb_t01)
        return scale_tensor_01_to_m11(gray_t01)

    raise ValueError(f"Unhandled in_channels={ic}.")


def extract_depth_m11_from_sample(sample_chw: torch.Tensor, out_channels: int) -> torch.Tensor:
    oc = int(out_channels)
    if sample_chw.ndim != 3:
        raise ValueError(f"Expected sample_chw [C,H,W], got {tuple(sample_chw.shape)}")

    if oc == 4:
        if sample_chw.shape[0] < 4:
            raise ValueError(f"Expected >=4 channels in sample for out_channels=4, got {sample_chw.shape[0]}")
        return sample_chw[3:4]
    if oc == 1:
        if sample_chw.shape[0] < 1:
            raise ValueError("Sample has no channels.")
        return sample_chw[0:1]
    raise ValueError(f"Unsupported out_channels={oc}. Expected 1 or 4.")


def extract_rgb_m11_from_sample(sample_chw: torch.Tensor, out_channels: int) -> Optional[torch.Tensor]:
    """
    If model outputs RGBD (out_channels=4), return RGB in [-1,1] as [3,H,W].
    Else return None.
    """
    oc = int(out_channels)
    if oc != 4:
        return None
    if sample_chw.ndim != 3 or sample_chw.shape[0] < 3:
        return None
    return sample_chw[0:3]


# =============================================================================
# PRED SAVE/LOAD (depth: uint16 0..1, rgb: uint8 0..255)
# =============================================================================
def save_pred01_as_u16(pred01_hw: torch.Tensor, out_path: str):
    p = pred01_hw.detach().cpu().float().clamp(0.0, 1.0)
    arr = p.numpy()
    u16 = np.round(arr * 65535.0).astype(np.uint16)
    Image.fromarray(u16, mode="I;16").save(out_path)


def load_pred_u16_to_pred01(pred_path: str, input_size: int) -> torch.Tensor:
    img = Image.open(pred_path)
    arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.float32)

    if arr.max() <= 255.0 + 1e-6:
        pred01 = arr / 255.0
    else:
        pred01 = arr / 65535.0

    t = torch.from_numpy(pred01).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(input_size, input_size), mode="nearest")
    return t[0, 0]


def save_rgb01_as_u8(rgb01_chw: torch.Tensor, out_path: str):
    """
    rgb01_chw: [3,H,W] in [0,1]
    """
    rgb = rgb01_chw.detach().cpu().float().clamp(0.0, 1.0)
    arr = (rgb.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)  # [H,W,3]
    Image.fromarray(arr, mode="RGB").save(out_path)


def save_depth_vis(pred01_hw: torch.Tensor, out_path: str, cmap_name: str = "magma"):
    """
    Saves a readable depth visualization (colored) as png.
    """
    p = pred01_hw.detach().cpu().float().clamp(0.0, 1.0).numpy()
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(p, cmap=cmap_name, vmin=0.0, vmax=1.0)
    ax.axis("off")
    fig.tight_layout(pad=0.0)
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def _should_load_existing_depth(depth_path: str, args) -> bool:
    if not os.path.isfile(depth_path):
        return False
    if bool(args.overwrite_pred_depth):
        return False
    return bool(args.reuse_pred_depth)


def _missing_any_outputs(depth_path: str, rgb_path: str, vis_path: str, need_rgb: bool) -> bool:
    if not os.path.isfile(depth_path):
        return True
    if not os.path.isfile(vis_path):
        return True
    if need_rgb and (not os.path.isfile(rgb_path)):
        return True
    return False


@torch.no_grad()
def get_or_make_pred_outputs(
    *,
    img_path: str,
    stem: str,
    pred_depth_path: str,
    pred_rgb_path: str,
    pred_vis_path: str,
    args,
    seed: int,
    diffusion=None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns:
      pred_depth01_map: [S,S] float in [0,1]
      pred_rgb01_map (optional): [3,S,S] float in [0,1] if model outputs RGBD and saving enabled
    """
    need_rgb = (int(args.out_channels) == 4) and bool(args.save_pred_rgb)

    # Reuse depth if present (fast path) unless we must regenerate to fill missing outputs
    if _should_load_existing_depth(pred_depth_path, args):
        pred01 = load_pred_u16_to_pred01(pred_depth_path, args.input_size)

        # Save vis if missing
        if bool(args.save_pred_vis) and (not os.path.isfile(pred_vis_path)):
            save_depth_vis(pred01, pred_vis_path, cmap_name=str(args.depth_vis_cmap))

        # RGB cannot be reconstructed from depth; optionally force regenerate if missing and requested
        if need_rgb and (not os.path.isfile(pred_rgb_path)) and bool(args.overwrite_missing_outputs):
            # fall through to regenerate
            pass
        else:
            return pred01, None

    # If we reached here, we must sample
    if diffusion is None or device is None:
        raise RuntimeError("Need diffusion+device to infer prediction, but they were not provided.")

    img_seed = per_image_seed(seed, stem)
    set_all_seeds(img_seed)

    cond = build_condition_tensor(img_path, args.input_size, args.in_channels, args).unsqueeze(0).to(device)
    sample = diffusion.sample(batch_size=1, condition_tensors=cond).detach().cpu()[0]  # [C,H,W] in [-1,1]

    depth_m11 = extract_depth_m11_from_sample(sample, out_channels=int(args.out_channels))  # [1,H,W]
    pred_depth01 = ((depth_m11 + 1.0) * 0.5).clamp(0.0, 1.0)[0]  # [H,W]

    pred_rgb01: Optional[torch.Tensor] = None
    if int(args.out_channels) == 4:
        rgb_m11 = extract_rgb_m11_from_sample(sample, out_channels=int(args.out_channels))
        if rgb_m11 is not None:
            pred_rgb01 = ((rgb_m11 + 1.0) * 0.5).clamp(0.0, 1.0)  # [3,H,W]

    # Save
    if bool(args.save_pred_depth):
        ensure_dir(os.path.dirname(pred_depth_path))
        save_pred01_as_u16(pred_depth01, pred_depth_path)

    if bool(args.save_pred_vis):
        save_depth_vis(pred_depth01, pred_vis_path, cmap_name=str(args.depth_vis_cmap))

    if need_rgb and (pred_rgb01 is not None):
        ensure_dir(os.path.dirname(pred_rgb_path))
        save_rgb01_as_u8(pred_rgb01, pred_rgb_path)

    return pred_depth01, pred_rgb01


# =============================================================================
# WEIGHT SPEC PARSING + RESOLUTION (kept compatible)
# =============================================================================
@dataclass
class WeightSpec:
    weight_input: str
    unet_name: str
    path: str
    input_size: Optional[int] = None
    attention_resolutions: Optional[str] = None
    in_channels: Optional[int] = None


def _parse_unet_overrides(left: str) -> Tuple[str, Optional[int], Optional[str], Optional[int]]:
    left = str(left or "").strip()
    if not left:
        return "", None, None, None

    parts = [p.strip() for p in left.split("@") if p.strip() != ""]
    if not parts:
        return "", None, None, None

    unet_name = parts[0].strip()
    input_size: Optional[int] = None
    attn: Optional[str] = None
    in_channels: Optional[int] = None

    def _maybe_parse_in_channels(tok: str) -> Optional[int]:
        t = str(tok).strip().lower()
        if t in ("7ch", "7"):
            return 7
        if t in ("8ch", "8"):
            return 8
        if t == "5ch":
            return 5
        if t == "4ch":
            return 4
        if t == "2ch":
            return 2
        return None

    for tok in parts[1:]:
        ic = _maybe_parse_in_channels(tok)
        if ic is not None:
            in_channels = int(ic)
            continue

        if input_size is None:
            try:
                input_size = int(tok)
                continue
            except Exception:
                pass

        if attn is None:
            attn = normalize_attention_resolutions(tok)
            continue

    return unet_name, input_size, attn, in_channels


def parse_weight_spec(token: str, default_unet: str) -> WeightSpec:
    t = (token or "").strip()
    if not t:
        raise ValueError("Empty weight spec token.")

    if "::" in t:
        left, right = t.split("::", 1)
        left = left.strip()
        right = right.strip()
        if left:
            unet_name, isz, attn, in_ch = _parse_unet_overrides(left)
            base_unet = unet_name.split("@", 1)[0].strip() if unet_name else ""
            if unet_name and (base_unet in UNET_NAME_TO_MODULE):
                return WeightSpec(t, unet_name, right, isz, attn, in_ch)

    if ":" in t:
        left, right = t.split(":", 1)
        left_s = left.strip()
        right_s = right.strip()

        base_unet = left_s.split("@", 1)[0].strip()
        if base_unet in UNET_NAME_TO_MODULE:
            unet_name, isz, attn, in_ch = _parse_unet_overrides(left_s)
            return WeightSpec(t, unet_name, right_s, isz, attn, in_ch)

        return WeightSpec(t, default_unet, t, None, None, None)

    return WeightSpec(t, default_unet, t, None, None, None)


_MODEL_RE = re.compile(r"model-(\d+)\.pt$")


@dataclass
class ResolvedWeight:
    weight_input: str
    unet_name: str
    input_size: int
    attention_resolutions: str
    in_channels: int
    folder_path: str
    original_checkpoint_path: str
    checkpoint_path: str
    model_file: str
    checkpoint_step: int
    resolution_msg: str


def _parse_step_from_model_file(model_file: str) -> int:
    m = _MODEL_RE.search(str(model_file).strip())
    if not m:
        return -1
    try:
        return int(m.group(1))
    except Exception:
        return -1


def _atomic_copy2(src: str, dst: str):
    ensure_dir(os.path.dirname(dst))
    tmp = dst + f".tmp_{os.getpid()}_{int(time.time()*1000)}"
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def copy_checkpoint_to_bench(src_ckpt: str, bench_dir: str, unet_name: str, folder_path: str, in_channels: int) -> str:
    ensure_dir(bench_dir)
    base = os.path.basename(src_ckpt)
    folder_base = os.path.basename(folder_path.rstrip("/")) or "folder"
    dst_name = f"{unet_name}__ch{int(in_channels)}__{folder_base}__{base}"
    dst_path = os.path.join(bench_dir, dst_name)

    try:
        if os.path.isfile(dst_path):
            src_stat = os.stat(src_ckpt)
            dst_stat = os.stat(dst_path)
            if src_stat.st_size == dst_stat.st_size:
                return dst_path
    except Exception:
        pass

    _atomic_copy2(src_ckpt, dst_path)
    return dst_path


def resolve_weight_path(
    weight_spec: WeightSpec,
    bench_dir: str,
    default_input_size: int,
    default_attention_resolutions: str,
    default_in_channels: int,
) -> ResolvedWeight:
    p = weight_spec.path.strip()
    unet_name = weight_spec.unet_name.strip()

    eff_input_size = int(weight_spec.input_size) if (weight_spec.input_size is not None) else int(default_input_size)
    eff_attn = str(weight_spec.attention_resolutions).strip() if (weight_spec.attention_resolutions is not None) else str(default_attention_resolutions).strip()
    eff_attn = normalize_attention_resolutions(eff_attn) or str(default_attention_resolutions).strip()

    eff_in_channels = int(weight_spec.in_channels) if (weight_spec.in_channels is not None) else int(default_in_channels)

    if eff_in_channels not in (2, 4, 5, 7, 8):
        raise ValueError(f"Resolved in_channels must be one of 2,4,5,7,8, got {eff_in_channels} from spec={weight_spec.weight_input}")

    if unet_name not in UNET_NAME_TO_MODULE:
        known = ", ".join(sorted(UNET_NAME_TO_MODULE.keys()))
        raise ValueError(f"Unknown unet_name='{unet_name}' in spec '{weight_spec.weight_input}'. Known: {known}")

    if os.path.isfile(p):
        base = os.path.basename(p)
        step = _parse_step_from_model_file(base)
        msg = f"File provided: {p}"
        safe_path = copy_checkpoint_to_bench(p, bench_dir, unet_name, os.path.dirname(p), eff_in_channels)
        return ResolvedWeight(
            weight_spec.weight_input,
            unet_name,
            eff_input_size,
            eff_attn,
            eff_in_channels,
            os.path.dirname(p),
            p,
            safe_path,
            base,
            step,
            msg,
        )

    if os.path.isdir(p):
        cands = []
        for f in os.listdir(p):
            m = _MODEL_RE.search(f)
            if m:
                cands.append((int(m.group(1)), os.path.join(p, f)))
        if not cands:
            raise FileNotFoundError(f"No model-#.pt found in directory: {p}")
        cands.sort(key=lambda x: x[0])
        latest_num, latest_path = cands[-1]
        msg = f"Directory provided: {p} -> selected latest: {latest_path}"
        safe_path = copy_checkpoint_to_bench(latest_path, bench_dir, unet_name, p, eff_in_channels)
        return ResolvedWeight(
            weight_spec.weight_input,
            unet_name,
            eff_input_size,
            eff_attn,
            eff_in_channels,
            p,
            latest_path,
            safe_path,
            os.path.basename(latest_path),
            int(latest_num),
            msg,
        )

    raise FileNotFoundError(f"Weight path not found: {p} (from spec: {weight_spec.weight_input})")


def model_tag_from_resolved(rw: ResolvedWeight, n_eval_images: int) -> str:
    folder_base = os.path.basename(rw.folder_path.rstrip("/"))
    folder_base_eval = f"{folder_base}_{int(n_eval_images)}"
    attn_tag = slugify(rw.attention_resolutions, max_len=60)
    return f"{rw.unet_name}__ch{int(rw.in_channels)}__sz{int(rw.input_size)}__attn{attn_tag}__{folder_base_eval}__{rw.model_file}"


# =============================================================================
# MODEL LOADING
# =============================================================================
def load_diffusion_and_model(args, device: torch.device):
    create_model = get_create_model(args.unet_name)
    GaussianDiffusion = get_gaussian_diffusion_class(int(args.in_channels))

    unet = create_model(
        image_size=args.input_size,
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        attention_resolutions=args.attention_resolutions,
        dropout=args.dropout,
        use_scale_shift_norm=args.use_scale_shift_norm,
    ).to(device)

    diffusion = GaussianDiffusion(
        unet,
        image_size=args.input_size,
        channels=args.out_channels,
        timesteps=args.timesteps,
        loss_type=args.loss_type,
    ).to(device)

    diffusion.eval()
    unet.eval()

    print(f">> Loading checkpoint from (SAFE): {args.weight}")
    print(
        f">> UNet={args.unet_name} | in_ch={args.in_channels} | out_ch={args.out_channels} | "
        f"input_size={args.input_size} | attn={args.attention_resolutions}"
    )

    try:
        checkpoint = torch.load(args.weight, weights_only=True, map_location=device)
    except TypeError:
        checkpoint = torch.load(args.weight, map_location=device)

    if isinstance(checkpoint, dict):
        if bool(args.use_ema) and ("ema" in checkpoint):
            print("   Using EMA weights ('ema' key).")
            src_state = checkpoint["ema"]
        elif "model" in checkpoint:
            print("   Using 'model' weights.")
            src_state = checkpoint["model"]
        else:
            print("   Using checkpoint as raw state_dict.")
            src_state = checkpoint
    else:
        print("   Using checkpoint as raw state_dict.")
        src_state = checkpoint

    model_state = diffusion.state_dict()
    filtered_state = {k: v for k, v in src_state.items() if (k in model_state and model_state[k].shape == v.shape)}
    model_state.update(filtered_state)
    diffusion.load_state_dict(model_state)

    print(f"   Loaded {len(filtered_state)} tensors.")
    skipped = [k for k in src_state.keys() if k not in filtered_state]
    if skipped:
        print(f"   [Info] Skipped {len(skipped)} keys due to mismatch/unexpected.")

    return diffusion


# =============================================================================
# COORDINATE MAPPING
# =============================================================================
def map_point_to_resized_square(x: float, y: float, w: int, h: int, size: int) -> Tuple[int, int]:
    """
    Map (x,y) in original image coords to resized square coords.
      x' = round( x/(W-1) * (S-1) )
      y' = round( y/(H-1) * (S-1) )
    Then clamp to [0..S-1].
    """
    W = max(1, int(w))
    H = max(1, int(h))
    S = max(1, int(size))

    if W == 1:
        xr = 0
    else:
        xr = int(round(float(x) / float(W - 1) * float(S - 1)))

    if H == 1:
        yr = 0
    else:
        yr = int(round(float(y) / float(H - 1) * float(S - 1)))

    xr = clamp_int(xr, 0, S - 1)
    yr = clamp_int(yr, 0, S - 1)
    return xr, yr


# =============================================================================
# RANKING METRICS
# =============================================================================
@dataclass
class PerImageRankingMetrics:
    point_rank_accuracy: float
    correct_count_out_of_5: int
    exact_match: float
    pairwise_accuracy: float
    kendall_tau: float
    spearman_rho: float
    correct_points: int
    total_points: int
    correct_pairs: int
    total_pairs: int


def _spearman_rho_from_lists(gt_ranks: List[int], pred_vals: List[float]) -> float:
    """
    Spearman rho between GT ranks (integers) and predicted values.
    Pred values are ranked with average-rank for ties.
    """
    n = len(gt_ranks)
    if n <= 1:
        return float("nan")

    idx = list(range(n))
    idx.sort(key=lambda i: pred_vals[i])

    pred_rank = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(pred_vals[idx[j + 1]] - pred_vals[idx[i]]) <= 0.0:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            pred_rank[idx[k]] = avg_rank
        i = j + 1

    x = np.array(gt_ranks, dtype=np.float64)
    y = np.array(pred_rank, dtype=np.float64)

    x_mean = float(x.mean())
    y_mean = float(y.mean())
    x_c = x - x_mean
    y_c = y - y_mean
    denom = float(np.sqrt((x_c * x_c).sum() * (y_c * y_c).sum()))
    if denom <= 1e-12:
        return float("nan")
    return float((x_c * y_c).sum() / denom)


def compute_ranking_metrics_for_image(
    *,
    gt_points: List[AnnPoint],
    pred01_map: torch.Tensor,  # [S,S] in [0,1]
    orig_w: int,
    orig_h: int,
    input_size: int,
    rank1_is_closest: bool,
    pred_higher_is_farther: bool,
    tie_epsilon: float,
) -> Tuple[PerImageRankingMetrics, Dict[int, float], Dict[int, int], Dict[int, int], Dict[int, Tuple[int, int]]]:
    """
    Returns:
      metrics,
      pred_values_by_point_id,
      pred_ranks_by_point_id,
      gt_ranks_by_point_id,
      mapped_xy_by_point_id
    """
    # Define a "key" so sorting ascending yields predicted rank 1..N in the GT meaning.
    # rank1_is_closest:
    #   pred_higher_is_farther: closest has smaller value -> key=+value
    #   else: closest has larger value -> key=-value
    # rank1_is_farthest:
    #   pred_higher_is_farther: farthest has larger value -> key=-value
    #   else: farthest has smaller value -> key=+value
    if rank1_is_closest:
        key_sign = +1.0 if pred_higher_is_farther else -1.0
    else:
        key_sign = -1.0 if pred_higher_is_farther else +1.0

    pred_values: Dict[int, float] = {}
    gt_ranks: Dict[int, int] = {}
    mapped_xy: Dict[int, Tuple[int, int]] = {}

    for p in gt_points:
        xr, yr = map_point_to_resized_square(p.x, p.y, orig_w, orig_h, input_size)
        mapped_xy[p.point_id] = (xr, yr)
        v = float(pred01_map[yr, xr].item())
        pred_values[p.point_id] = v
        gt_ranks[p.point_id] = int(p.rank)

    point_ids = [p.point_id for p in gt_points]
    point_ids_sorted = sorted(point_ids, key=lambda pid: (key_sign * pred_values[pid], pid))

    pred_ranks: Dict[int, int] = {}
    for i, pid in enumerate(point_ids_sorted):
        pred_ranks[pid] = i + 1

    total_points = len(point_ids)
    correct_points = sum(1 for pid in point_ids if pred_ranks.get(pid, -999) == gt_ranks.get(pid, -888))
    point_rank_accuracy = float(correct_points) / float(total_points) if total_points > 0 else float("nan")

    # Out-of-5 bucket (as requested)
    correct_count_out_of_5 = int(correct_points)  # if total_points != 5 it is still "out of total_points", but most data is 5.

    exact_match = 1.0 if (correct_points == total_points and total_points > 0) else 0.0

    # Pairwise accuracy + Kendall tau (skip GT ties)
    total_pairs = 0
    correct_pairs = 0
    concordant = 0
    discordant = 0

    for i in range(len(point_ids)):
        for j in range(i + 1, len(point_ids)):
            pid_i = point_ids[i]
            pid_j = point_ids[j]
            gi = gt_ranks[pid_i]
            gj = gt_ranks[pid_j]
            if gi == gj:
                continue

            total_pairs += 1
            gt_i_better = gi < gj

            vi = key_sign * pred_values[pid_i]
            vj = key_sign * pred_values[pid_j]

            if abs(vi - vj) <= float(tie_epsilon):
                # count as discordant/incorrect
                discordant += 1
                continue

            pred_i_better = vi < vj
            if pred_i_better == gt_i_better:
                correct_pairs += 1
                concordant += 1
            else:
                discordant += 1

    pairwise_accuracy = float(correct_pairs) / float(total_pairs) if total_pairs > 0 else float("nan")
    kendall_tau = float(concordant - discordant) / float(total_pairs) if total_pairs > 0 else float("nan")

    # Spearman rho between GT ranks and predicted key-values
    gt_rank_list = [gt_ranks[pid] for pid in point_ids]
    pred_list = [key_sign * pred_values[pid] for pid in point_ids]
    spearman_rho = _spearman_rho_from_lists(gt_rank_list, pred_list)

    metrics = PerImageRankingMetrics(
        point_rank_accuracy=float(point_rank_accuracy),
        correct_count_out_of_5=int(correct_count_out_of_5),
        exact_match=float(exact_match),
        pairwise_accuracy=float(pairwise_accuracy),
        kendall_tau=float(kendall_tau),
        spearman_rho=float(spearman_rho),
        correct_points=int(correct_points),
        total_points=int(total_points),
        correct_pairs=int(correct_pairs),
        total_pairs=int(total_pairs),
    )
    return metrics, pred_values, pred_ranks, gt_ranks, mapped_xy


# =============================================================================
# LABELS FOR PLOTS
# =============================================================================
def parse_label_rules(label_args: List[str]) -> List[Tuple[str, str]]:
    rules: List[Tuple[str, str]] = []
    for item in (label_args or []):
        s = str(item)
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k and v:
            rules.append((k, v))
    return rules


def label_for_model(rw: ResolvedWeight, label_rules: List[Tuple[str, str]]) -> str:
    hay = " | ".join([rw.folder_path, rw.model_file, rw.weight_input])
    for match, label in label_rules:
        if match in hay:
            return label
    return f"{int(rw.in_channels)}ch"


def uniquify_labels(labels: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for lab in labels:
        if lab not in seen:
            seen[lab] = 1
            out.append(lab)
        else:
            seen[lab] += 1
            out.append(f"{lab} #{seen[lab]}")
    return out


# =============================================================================
# CSV FIELDS
# =============================================================================
MAIN_CSV_FIELDS = [
    "weight_input",
    "folder_path",
    "model_file",
    "checkpoint_step",
    "unet_name",
    "in_channels",
    "out_channels",
    "input_size",
    "attention_resolutions",
    "images_root",
    "edges_root",
    "annotation_json",
    "n_images",
    "seed",
    # main metrics
    "point_acc_overall",
    "mean_point_rank_accuracy",
    "exact_match_rate",
    "pairwise_accuracy_mean",
    "kendall_tau_mean",
    "spearman_rho_mean",
    # distribution "out of 5"
    "n_img_correct_5",
    "n_img_correct_4",
    "n_img_correct_3",
    "n_img_correct_2",
    "n_img_correct_1",
    "n_img_correct_0",
]

PER_IMAGE_CSV_FIELDS = [
    "weight_input",
    "folder_path",
    "model_file",
    "checkpoint_step",
    "unet_name",
    "in_channels",
    "out_channels",
    "input_size",
    "attention_resolutions",
    "images_root",
    "edges_root",
    "annotation_json",
    "seed",
    "filename",
    # metrics
    "point_rank_accuracy",
    "correct_count_out_of_5",
    "exact_match",
    "pairwise_accuracy",
    "kendall_tau",
    "spearman_rho",
    # point details (robust: store by sorted point_id order, up to 5)
    "pid_1",
    "pid_2",
    "pid_3",
    "pid_4",
    "pid_5",
    "gt_rank_1",
    "gt_rank_2",
    "gt_rank_3",
    "gt_rank_4",
    "gt_rank_5",
    "pred_rank_1",
    "pred_rank_2",
    "pred_rank_3",
    "pred_rank_4",
    "pred_rank_5",
    "pred_val_1",
    "pred_val_2",
    "pred_val_3",
    "pred_val_4",
    "pred_val_5",
    "mapped_x_1",
    "mapped_y_1",
    "mapped_x_2",
    "mapped_y_2",
    "mapped_x_3",
    "mapped_y_3",
    "mapped_x_4",
    "mapped_y_4",
    "mapped_x_5",
    "mapped_y_5",
]

SUMMARY_CSV_FIELDS = [
    "label",
    "weight_input",
    "folder_path",
    "model_file",
    "checkpoint_step",
    "unet_name",
    "in_channels",
    "out_channels",
    "input_size",
    "attention_resolutions",
    "n_seeds",
    "n_images",
    # mean±std across seeds
    "point_acc_overall_mean",
    "point_acc_overall_std",
    "mean_point_rank_accuracy_mean",
    "mean_point_rank_accuracy_std",
    "exact_match_rate_mean",
    "exact_match_rate_std",
    "pairwise_accuracy_mean",
    "pairwise_accuracy_std",
    "kendall_tau_mean",
    "kendall_tau_std",
    "spearman_rho_mean",
    "spearman_rho_std",
    # distribution across images (proportion mean±std across seeds)
    "prop_img_correct_5_mean",
    "prop_img_correct_5_std",
    "prop_img_correct_4_mean",
    "prop_img_correct_4_std",
    "prop_img_correct_3_mean",
    "prop_img_correct_3_std",
    "prop_img_correct_2_mean",
    "prop_img_correct_2_std",
    "prop_img_correct_1_mean",
    "prop_img_correct_1_std",
    "prop_img_correct_0_mean",
    "prop_img_correct_0_std",
]


def write_csv(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]):
    ensure_dir(os.path.dirname(path))
    tmp = path + f".tmp_{os.getpid()}_{int(time.time()*1000)}"
    try:
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if os.path.isfile(tmp):
                os.remove(tmp)
        except Exception:
            pass


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return float("nan")
        return v
    except Exception:
        return float("nan")


def _nanmean_std(vals: List[float]) -> Tuple[float, float]:
    v = [float(x) for x in vals if not (math.isnan(float(x)) or math.isinf(float(x)))]
    if len(v) == 0:
        return float("nan"), float("nan")
    if len(v) == 1:
        return float(v[0]), 0.0
    arr = np.array(v, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


# =============================================================================
# CORE EVALUATION
# =============================================================================
@torch.no_grad()
def eval_one_model_one_seed(
    *,
    args,
    rw: ResolvedWeight,
    diffusion,
    device: torch.device,
    seed: int,
    ann_images: List[AnnImage],
    model_out_dir: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    apply_torch_determinism(args)
    set_all_seeds(seed)

    out_dir_seed = os.path.join(model_out_dir, f"seed_{seed}")
    ensure_dir(out_dir_seed)

    pred_depth_dir = os.path.join(out_dir_seed, "pred_depth_u16")
    pred_vis_dir = os.path.join(out_dir_seed, "pred_depth_vis")
    pred_rgb_dir = os.path.join(out_dir_seed, "pred_rgb_u8")
    ensure_dir(pred_depth_dir)
    ensure_dir(pred_vis_dir)
    if bool(args.save_pred_rgb) and int(args.out_channels) == 4:
        ensure_dir(pred_rgb_dir)

    total_correct_points = 0
    total_points = 0

    sum_point_rank_acc = 0.0
    sum_exact = 0.0
    sum_pairwise = 0.0
    sum_tau = 0.0
    sum_rho = 0.0

    # out-of-5 distribution across images
    img_correct_count_bins = {k: 0 for k in range(0, 6)}  # 0..5

    per_image_rows: List[Dict[str, Any]] = []

    for ai in ann_images:
        stem = os.path.splitext(os.path.basename(ai.filename))[0]
        pred_depth_path = os.path.join(pred_depth_dir, f"{stem}.png")
        pred_vis_path = os.path.join(pred_vis_dir, f"{stem}.png")
        pred_rgb_path = os.path.join(pred_rgb_dir, f"{stem}.png")  # only used if out_channels=4

        pred_depth01, _pred_rgb01 = get_or_make_pred_outputs(
            img_path=ai.path,
            stem=stem,
            pred_depth_path=pred_depth_path,
            pred_rgb_path=pred_rgb_path,
            pred_vis_path=pred_vis_path,
            args=args,
            seed=seed,
            diffusion=diffusion,
            device=device,
        )

        metrics, pred_vals, pred_ranks, gt_ranks, mapped_xy = compute_ranking_metrics_for_image(
            gt_points=ai.points,
            pred01_map=pred_depth01,
            orig_w=ai.width,
            orig_h=ai.height,
            input_size=args.input_size,
            rank1_is_closest=bool(args.rank1_is_closest),
            pred_higher_is_farther=bool(args.pred_higher_is_farther),
            tie_epsilon=float(args.tie_epsilon),
        )

        total_correct_points += int(metrics.correct_points)
        total_points += int(metrics.total_points)

        sum_point_rank_acc += float(metrics.point_rank_accuracy)
        sum_exact += float(metrics.exact_match)
        sum_pairwise += float(metrics.pairwise_accuracy)
        sum_tau += float(metrics.kendall_tau)
        sum_rho += float(metrics.spearman_rho)

        # bin count (clamp just in case)
        cc = clamp_int(int(metrics.correct_count_out_of_5), 0, 5)
        img_correct_count_bins[cc] += 1

        # Per-image details stored in a stable order by sorted point_id
        pids_sorted = sorted([p.point_id for p in ai.points])
        pids_sorted = (pids_sorted + [None] * 5)[:5]

        def _pid(i: int) -> Any:
            return pids_sorted[i]

        def _get(d: Dict[int, Any], pid: Optional[int], default: Any) -> Any:
            if pid is None:
                return default
            return d.get(pid, default)

        def _get_xy(pid: Optional[int]) -> Tuple[Any, Any]:
            if pid is None:
                return ("", "")
            xy = mapped_xy.get(pid, None)
            if xy is None:
                return ("", "")
            return (int(xy[0]), int(xy[1]))

        row_img = dict(
            weight_input=str(rw.weight_input),
            folder_path=str(rw.folder_path),
            model_file=str(rw.model_file),
            checkpoint_step=int(rw.checkpoint_step),
            unet_name=str(rw.unet_name),
            in_channels=int(rw.in_channels),
            out_channels=int(args.out_channels),
            input_size=int(args.input_size),
            attention_resolutions=str(args.attention_resolutions),
            images_root=str(args.images_root),
            edges_root=str(args.edges_root),
            annotation_json=str(args.annotation_json),
            seed=int(seed),
            filename=str(ai.filename),
            point_rank_accuracy=float(metrics.point_rank_accuracy),
            correct_count_out_of_5=int(metrics.correct_count_out_of_5),
            exact_match=float(metrics.exact_match),
            pairwise_accuracy=float(metrics.pairwise_accuracy),
            kendall_tau=float(metrics.kendall_tau),
            spearman_rho=float(metrics.spearman_rho),
            pid_1=_pid(0) if _pid(0) is not None else "",
            pid_2=_pid(1) if _pid(1) is not None else "",
            pid_3=_pid(2) if _pid(2) is not None else "",
            pid_4=_pid(3) if _pid(3) is not None else "",
            pid_5=_pid(4) if _pid(4) is not None else "",
            gt_rank_1=_get(gt_ranks, _pid(0), ""),
            gt_rank_2=_get(gt_ranks, _pid(1), ""),
            gt_rank_3=_get(gt_ranks, _pid(2), ""),
            gt_rank_4=_get(gt_ranks, _pid(3), ""),
            gt_rank_5=_get(gt_ranks, _pid(4), ""),
            pred_rank_1=_get(pred_ranks, _pid(0), ""),
            pred_rank_2=_get(pred_ranks, _pid(1), ""),
            pred_rank_3=_get(pred_ranks, _pid(2), ""),
            pred_rank_4=_get(pred_ranks, _pid(3), ""),
            pred_rank_5=_get(pred_ranks, _pid(4), ""),
            pred_val_1=_get(pred_vals, _pid(0), ""),
            pred_val_2=_get(pred_vals, _pid(1), ""),
            pred_val_3=_get(pred_vals, _pid(2), ""),
            pred_val_4=_get(pred_vals, _pid(3), ""),
            pred_val_5=_get(pred_vals, _pid(4), ""),
            mapped_x_1=_get_xy(_pid(0))[0],
            mapped_y_1=_get_xy(_pid(0))[1],
            mapped_x_2=_get_xy(_pid(1))[0],
            mapped_y_2=_get_xy(_pid(1))[1],
            mapped_x_3=_get_xy(_pid(2))[0],
            mapped_y_3=_get_xy(_pid(2))[1],
            mapped_x_4=_get_xy(_pid(3))[0],
            mapped_y_4=_get_xy(_pid(3))[1],
            mapped_x_5=_get_xy(_pid(4))[0],
            mapped_y_5=_get_xy(_pid(4))[1],
        )
        per_image_rows.append(row_img)

    n_images = len(ann_images)
    if n_images <= 0:
        raise RuntimeError("No images evaluated (n_images=0).")

    point_acc_overall = float(total_correct_points) / float(total_points) if total_points > 0 else float("nan")
    mean_point_rank_accuracy = float(sum_point_rank_acc) / float(n_images)
    exact_match_rate = float(sum_exact) / float(n_images)
    pairwise_accuracy_mean = float(sum_pairwise) / float(n_images)
    kendall_tau_mean = float(sum_tau) / float(n_images)
    spearman_rho_mean = float(sum_rho) / float(n_images)

    main_row = dict(
        weight_input=str(rw.weight_input),
        folder_path=str(rw.folder_path),
        model_file=str(rw.model_file),
        checkpoint_step=int(rw.checkpoint_step),
        unet_name=str(rw.unet_name),
        in_channels=int(rw.in_channels),
        out_channels=int(args.out_channels),
        input_size=int(args.input_size),
        attention_resolutions=str(args.attention_resolutions),
        images_root=str(args.images_root),
        edges_root=str(args.edges_root),
        annotation_json=str(args.annotation_json),
        n_images=int(n_images),
        seed=int(seed),
        point_acc_overall=float(point_acc_overall),
        mean_point_rank_accuracy=float(mean_point_rank_accuracy),
        exact_match_rate=float(exact_match_rate),
        pairwise_accuracy_mean=float(pairwise_accuracy_mean),
        kendall_tau_mean=float(kendall_tau_mean),
        spearman_rho_mean=float(spearman_rho_mean),
        n_img_correct_5=int(img_correct_count_bins.get(5, 0)),
        n_img_correct_4=int(img_correct_count_bins.get(4, 0)),
        n_img_correct_3=int(img_correct_count_bins.get(3, 0)),
        n_img_correct_2=int(img_correct_count_bins.get(2, 0)),
        n_img_correct_1=int(img_correct_count_bins.get(1, 0)),
        n_img_correct_0=int(img_correct_count_bins.get(0, 0)),
    )

    return main_row, per_image_rows


# =============================================================================
# SUMMARY + PLOTS (READABLE)
# =============================================================================
def _plot_bars_with_err(
    labels: List[str],
    means: List[float],
    stds: List[float],
    title: str,
    ylabel: str,
    out_basepath_no_ext: str,
    formats: List[str],
    y_lim: Optional[Tuple[float, float]] = None,
    annotate: bool = True,
):
    x = np.arange(len(labels), dtype=np.float32)

    fig = plt.figure(figsize=(max(12, 0.65 * len(labels)), 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(x, means, yerr=stds, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])

    if annotate:
        for i, (m, s) in enumerate(zip(means, stds)):
            if math.isnan(m) or math.isinf(m):
                continue
            ax.text(i, m + (s if not math.isnan(s) else 0.0) + 0.01, f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    for fmt in formats:
        out_path = f"{out_basepath_no_ext}.{fmt}"
        fig.savefig(out_path, dpi=250)
        print(f"[PLOT] Saved: {out_path}")
    plt.close(fig)


def _plot_stacked_distribution(
    labels: List[str],
    props_by_k: Dict[int, List[float]],  # k -> list aligned with labels
    title: str,
    out_basepath_no_ext: str,
    formats: List[str],
):
    """
    Stacked bar: percentage of images with correct_count = 0..5.
    """
    x = np.arange(len(labels), dtype=np.float32)

    fig = plt.figure(figsize=(max(12, 0.70 * len(labels)), 6))
    ax = fig.add_subplot(1, 1, 1)

    bottom = np.zeros(len(labels), dtype=np.float32)
    # Plot from 0->5 so top segment corresponds to 5
    for k in range(0, 6):
        vals = np.array(props_by_k.get(k, [0.0] * len(labels)), dtype=np.float32)
        ax.bar(x, vals, bottom=bottom, label=f"{k}/5")
        bottom = bottom + vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Proportion of images")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(title="Correct points", ncol=3, fontsize=9)

    fig.tight_layout()
    for fmt in formats:
        out_path = f"{out_basepath_no_ext}.{fmt}"
        fig.savefig(out_path, dpi=250)
        print(f"[PLOT] Saved: {out_path}")
    plt.close(fig)


def generate_summary_and_plots(
    *,
    main_rows: List[Dict[str, Any]],
    summary_csv_path: str,
    plots_dir: str,
    plots_formats: List[str],
    label_rules: List[Tuple[str, str]],
):
    if not main_rows:
        print("[SUMMARY] No main rows; skipping summary/plots.")
        return

    # Group by model config (without seed)
    groups: Dict[Tuple[str, str, str, int, int, str], List[Dict[str, Any]]] = {}
    for r in main_rows:
        key = (
            str(r.get("weight_input", "")),
            str(r.get("folder_path", "")),
            str(r.get("model_file", "")),
            int(float(r.get("in_channels", 0) or 0)),
            int(float(r.get("input_size", 0) or 0)),
            str(r.get("attention_resolutions", "")),
        )
        groups.setdefault(key, []).append(r)

    summary_rows: List[Dict[str, Any]] = []
    raw_labels: List[str] = []
    model_keys_in_order: List[Tuple[str, str, str, int, int, str]] = []

    for key, gr in groups.items():
        weight_input, folder_path, model_file, in_ch, input_size, attn = key
        first = gr[0]

        rw_like = ResolvedWeight(
            weight_input=weight_input,
            unet_name=str(first.get("unet_name", "")),
            input_size=int(input_size),
            attention_resolutions=str(attn),
            in_channels=int(in_ch),
            folder_path=str(folder_path),
            original_checkpoint_path="",
            checkpoint_path="",
            model_file=str(model_file),
            checkpoint_step=int(float(first.get("checkpoint_step", -1) or -1)),
            resolution_msg="",
        )
        label = label_for_model(rw_like, label_rules)

        seeds = sorted({int(float(x.get("seed", 0) or 0)) for x in gr})
        n_images = int(float(first.get("n_images", 0) or 0))

        # Main metrics across seeds
        point_overall = [_safe_float(x.get("point_acc_overall", "nan")) for x in gr]
        mean_point = [_safe_float(x.get("mean_point_rank_accuracy", "nan")) for x in gr]
        exact = [_safe_float(x.get("exact_match_rate", "nan")) for x in gr]
        pairwise = [_safe_float(x.get("pairwise_accuracy_mean", "nan")) for x in gr]
        tau = [_safe_float(x.get("kendall_tau_mean", "nan")) for x in gr]
        rho = [_safe_float(x.get("spearman_rho_mean", "nan")) for x in gr]

        po_m, po_s = _nanmean_std(point_overall)
        mp_m, mp_s = _nanmean_std(mean_point)
        ex_m, ex_s = _nanmean_std(exact)
        pw_m, pw_s = _nanmean_std(pairwise)
        kt_m, kt_s = _nanmean_std(tau)
        sr_m, sr_s = _nanmean_std(rho)

        # Distribution proportions per seed, then mean/std
        prop_lists: Dict[int, List[float]] = {k: [] for k in range(0, 6)}
        for rr in gr:
            nimg = int(float(rr.get("n_images", 0) or 0))
            if nimg <= 0:
                continue
            for k in range(0, 6):
                c = float(rr.get(f"n_img_correct_{k}", 0) or 0)
                prop_lists[k].append(c / float(nimg))

        prop_stats: Dict[int, Tuple[float, float]] = {}
        for k in range(0, 6):
            prop_stats[k] = _nanmean_std(prop_lists[k])

        summary_rows.append(
            dict(
                label=label,
                weight_input=weight_input,
                folder_path=folder_path,
                model_file=model_file,
                checkpoint_step=int(float(first.get("checkpoint_step", -1) or -1)),
                unet_name=str(first.get("unet_name", "")),
                in_channels=int(in_ch),
                out_channels=int(float(first.get("out_channels", 0) or 0)),
                input_size=int(input_size),
                attention_resolutions=str(attn),
                n_seeds=int(len(seeds)),
                n_images=int(n_images),
                point_acc_overall_mean=float(po_m),
                point_acc_overall_std=float(po_s),
                mean_point_rank_accuracy_mean=float(mp_m),
                mean_point_rank_accuracy_std=float(mp_s),
                exact_match_rate_mean=float(ex_m),
                exact_match_rate_std=float(ex_s),
                pairwise_accuracy_mean=float(pw_m),
                pairwise_accuracy_std=float(pw_s),
                kendall_tau_mean=float(kt_m),
                kendall_tau_std=float(kt_s),
                spearman_rho_mean=float(sr_m),
                spearman_rho_std=float(sr_s),
                prop_img_correct_5_mean=float(prop_stats[5][0]),
                prop_img_correct_5_std=float(prop_stats[5][1]),
                prop_img_correct_4_mean=float(prop_stats[4][0]),
                prop_img_correct_4_std=float(prop_stats[4][1]),
                prop_img_correct_3_mean=float(prop_stats[3][0]),
                prop_img_correct_3_std=float(prop_stats[3][1]),
                prop_img_correct_2_mean=float(prop_stats[2][0]),
                prop_img_correct_2_std=float(prop_stats[2][1]),
                prop_img_correct_1_mean=float(prop_stats[1][0]),
                prop_img_correct_1_std=float(prop_stats[1][1]),
                prop_img_correct_0_mean=float(prop_stats[0][0]),
                prop_img_correct_0_std=float(prop_stats[0][1]),
            )
        )
        raw_labels.append(label)
        model_keys_in_order.append(key)

    # Make labels unique
    unique_labels = uniquify_labels(raw_labels)
    for i in range(len(summary_rows)):
        summary_rows[i]["label"] = unique_labels[i]

    # Sort by mean_point_rank_accuracy_mean (descending)
    def _sort_key(r: Dict[str, Any]):
        v = _safe_float(r.get("mean_point_rank_accuracy_mean", float("nan")))
        if math.isnan(v) or math.isinf(v):
            return (1, -1e9, str(r.get("label", "")))
        return (0, -v, str(r.get("label", "")))

    summary_rows.sort(key=_sort_key)

    # Write summary CSV
    write_csv(summary_csv_path, SUMMARY_CSV_FIELDS, summary_rows)
    print(f"[SUMMARY] Wrote: {summary_csv_path}")

    # Plots
    ensure_dir(plots_dir)
    labels_sorted = [r["label"] for r in summary_rows]

    # Main metric bars
    _plot_bars_with_err(
        labels=labels_sorted,
        means=[_safe_float(r.get("mean_point_rank_accuracy_mean", "nan")) for r in summary_rows],
        stds=[_safe_float(r.get("mean_point_rank_accuracy_std", "nan")) for r in summary_rows],
        title="Mean Point Rank Accuracy (fraction correct ranks among 5 points) — mean±std across seeds",
        ylabel="Accuracy (0..1)",
        out_basepath_no_ext=os.path.join(plots_dir, "mean_point_rank_accuracy"),
        formats=plots_formats,
        y_lim=(0.0, 1.0),
        annotate=True,
    )

    _plot_bars_with_err(
        labels=labels_sorted,
        means=[_safe_float(r.get("pairwise_accuracy_mean", "nan")) for r in summary_rows],
        stds=[_safe_float(r.get("pairwise_accuracy_std", "nan")) for r in summary_rows],
        title="Pairwise Ordering Accuracy (10 pairs) — mean±std across seeds",
        ylabel="Accuracy (0..1)",
        out_basepath_no_ext=os.path.join(plots_dir, "pairwise_accuracy"),
        formats=plots_formats,
        y_lim=(0.0, 1.0),
        annotate=True,
    )

    _plot_bars_with_err(
        labels=labels_sorted,
        means=[_safe_float(r.get("exact_match_rate_mean", "nan")) for r in summary_rows],
        stds=[_safe_float(r.get("exact_match_rate_std", "nan")) for r in summary_rows],
        title="Exact Match Rate (all 5 ranks correct) — mean±std across seeds",
        ylabel="Rate (0..1)",
        out_basepath_no_ext=os.path.join(plots_dir, "exact_match_rate"),
        formats=plots_formats,
        y_lim=(0.0, 1.0),
        annotate=True,
    )

    _plot_bars_with_err(
        labels=labels_sorted,
        means=[_safe_float(r.get("kendall_tau_mean", "nan")) for r in summary_rows],
        stds=[_safe_float(r.get("kendall_tau_std", "nan")) for r in summary_rows],
        title="Kendall Tau — mean±std across seeds",
        ylabel="Tau (-1..1)",
        out_basepath_no_ext=os.path.join(plots_dir, "kendall_tau"),
        formats=plots_formats,
        y_lim=(-1.0, 1.0),
        annotate=True,
    )

    _plot_bars_with_err(
        labels=labels_sorted,
        means=[_safe_float(r.get("spearman_rho_mean", "nan")) for r in summary_rows],
        stds=[_safe_float(r.get("spearman_rho_std", "nan")) for r in summary_rows],
        title="Spearman Rho — mean±std across seeds",
        ylabel="Rho (-1..1)",
        out_basepath_no_ext=os.path.join(plots_dir, "spearman_rho"),
        formats=plots_formats,
        y_lim=(-1.0, 1.0),
        annotate=True,
    )

    # Stacked distribution: proportions for correct_count_out_of_5 buckets
    props_by_k: Dict[int, List[float]] = {k: [] for k in range(0, 6)}
    for r in summary_rows:
        for k in range(0, 6):
            props_by_k[k].append(_safe_float(r.get(f"prop_img_correct_{k}_mean", 0.0)))

    _plot_stacked_distribution(
        labels=labels_sorted,
        props_by_k=props_by_k,
        title="Distribution of Correct Ranks per Image (out of 5 points) — mean proportion across seeds",
        out_basepath_no_ext=os.path.join(plots_dir, "correct_count_distribution_out_of_5"),
        formats=plots_formats,
    )


# =============================================================================
# BENCHMARK DRIVER
# =============================================================================
def run_benchmark(args):
    print(f">> CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}")
    n_gpu = torch.cuda.device_count()
    print(f">> torch sees {n_gpu} GPU(s)")

    if not os.path.isdir(args.images_root):
        raise FileNotFoundError(f"--images_root not found: {args.images_root}")
    if not os.path.isfile(args.annotation_json):
        raise FileNotFoundError(f"--annotation_json not found: {args.annotation_json}")

    ann_images = load_annotation_json(args.annotation_json, args.images_root, max_images=int(args.max_eval_images))
    n_eval = len(ann_images)
    print(f">> Annotated eval images: {n_eval} (max_eval_images={args.max_eval_images}; 0 => all annotated)")

    seeds = parse_seed_list(args.seed, args.seeds)
    print(f">> Seeds: {seeds}")

    ensure_dir(args.output_root)
    bench_models_dir = os.path.join(args.output_root, "bench_models")
    ensure_dir(bench_models_dir)

    main_csv_path = os.path.join(args.output_root, "ranking_metrics.csv")
    per_image_csv_path = os.path.join(args.output_root, "ranking_per_image.csv")
    summary_csv_path = os.path.join(args.output_root, "ranking_summary.csv")
    plots_dir = os.path.join(args.output_root, "plots")

    # Resolve weights
    if args.weights and len(args.weights) > 0:
        weight_tokens = args.weights
    elif args.weight and str(args.weight).strip():
        weight_tokens = [args.weight]
    else:
        weight_tokens = list(DEFAULT_WEIGHT_SPECS)

    weight_specs: List[WeightSpec] = [parse_weight_spec(t, args.default_unet) for t in weight_tokens]

    resolved: List[ResolvedWeight] = []
    print("\n==================== WEIGHT RESOLUTION (with SAFE COPY) ====================")
    for ws in weight_specs:
        rw = resolve_weight_path(
            ws,
            bench_dir=bench_models_dir,
            default_input_size=int(args.input_size),
            default_attention_resolutions=str(args.attention_resolutions),
            default_in_channels=int(args.in_channels),
        )
        resolved.append(rw)
        print(f"{rw.resolution_msg} | unet={rw.unet_name} | ch={rw.in_channels} | input_size={rw.input_size} | attn={rw.attention_resolutions}")
        print(f"   original: {rw.original_checkpoint_path}")
        print(f"   safe    : {rw.checkpoint_path}")
    print("============================================================================\n")

    any_8ch = any(int(rw.in_channels) == 8 for rw in resolved)
    if any_8ch:
        if not str(args.edges_root or "").strip():
            raise ValueError("At least one 8ch spec is requested but --edges_root is empty.")
        if not os.path.isdir(args.edges_root):
            raise FileNotFoundError(f"--edges_root not found (required for 8ch): {args.edges_root}")
    else:
        print(">> No 8ch specs detected -> --edges_root will not be used/validated.")

    # Device
    if args.cpu or (not torch.cuda.is_available()):
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{int(args.device)}")
    print(f">> Using device: {device}")

    label_rules = parse_label_rules(args.label)

    all_main_rows: List[Dict[str, Any]] = []
    all_per_image_rows: List[Dict[str, Any]] = []

    for rw in resolved:
        print("\n==================== MODEL ====================")
        print(f">> folder_path: {rw.folder_path}")
        print(f">> model_file : {rw.model_file}")
        print(f">> checkpoint_step: {rw.checkpoint_step}")
        print(f">> unet_name  : {rw.unet_name}")
        print(f">> in_channels: {rw.in_channels}")
        print(f">> input_size : {rw.input_size}")
        print(f">> attn_res   : {rw.attention_resolutions}")
        print(f">> checkpoint(SAFE): {rw.checkpoint_path}")
        print("==============================================\n")

        args_model = argparse.Namespace(**vars(args))
        args_model.weight = rw.checkpoint_path
        args_model.unet_name = rw.unet_name
        args_model.input_size = int(rw.input_size)
        args_model.attention_resolutions = str(rw.attention_resolutions)
        args_model.in_channels = int(rw.in_channels)
        args_model.out_channels = int(infer_out_channels_from_in_channels(int(rw.in_channels)))

        # If model outputs only depth, don't try to save rgb
        if int(args_model.out_channels) != 4:
            args_model.save_pred_rgb = False

        model_out_dir = os.path.join(args.output_root, model_tag_from_resolved(rw, n_eval))
        ensure_dir(model_out_dir)

        diffusion = load_diffusion_and_model(args_model, device)

        for s in seeds:
            print(f"\n>> Seed {s} | Evaluating {n_eval} images (ranking metrics)")
            main_row, per_image_rows = eval_one_model_one_seed(
                args=args_model,
                rw=rw,
                diffusion=diffusion,
                device=device,
                seed=int(s),
                ann_images=ann_images,
                model_out_dir=model_out_dir,
            )

            # Print concise per-seed summary including out-of-5 distribution
            dist = (
                f"out_of_5: "
                f"5:{main_row['n_img_correct_5']} "
                f"4:{main_row['n_img_correct_4']} "
                f"3:{main_row['n_img_correct_3']} "
                f"2:{main_row['n_img_correct_2']} "
                f"1:{main_row['n_img_correct_1']} "
                f"0:{main_row['n_img_correct_0']}"
            )
            print(
                f">> seed={s} | mean_point_rank_acc={main_row['mean_point_rank_accuracy']:.4f} | "
                f"pairwise_acc={main_row['pairwise_accuracy_mean']:.4f} | "
                f"exact_match={main_row['exact_match_rate']:.4f} | "
                f"kendall_tau={main_row['kendall_tau_mean']:.4f} | "
                f"spearman_rho={main_row['spearman_rho_mean']:.4f} | {dist}"
            )

            all_main_rows.append(main_row)
            all_per_image_rows.extend(per_image_rows)

            if bool(args.flush_csv_each_seed):
                write_csv(main_csv_path, MAIN_CSV_FIELDS, all_main_rows)
                write_csv(per_image_csv_path, PER_IMAGE_CSV_FIELDS, all_per_image_rows)

        del diffusion
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Final write
    write_csv(main_csv_path, MAIN_CSV_FIELDS, all_main_rows)
    write_csv(per_image_csv_path, PER_IMAGE_CSV_FIELDS, all_per_image_rows)
    print(f"\n[CSV] Wrote main:     {main_csv_path}")
    print(f"[CSV] Wrote per-img: {per_image_csv_path}")

    # Summary + plots
    if bool(args.make_plots) or bool(args.make_summary):
        plots_formats = [x.strip().lower() for x in str(args.plots_format).split(",") if x.strip()]
        plots_formats = [x for x in plots_formats if x in ("png", "pdf", "svg")]
        if not plots_formats:
            plots_formats = ["png"]

        generate_summary_and_plots(
            main_rows=all_main_rows,
            summary_csv_path=summary_csv_path,
            plots_dir=plots_dir,
            plots_formats=plots_formats,
            label_rules=label_rules,
        )


# =============================================================================
# ARGS
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Refined depth ranking benchmark (5 points per image) for Kvasir-SEG.")

    # Data
    parser.add_argument("--images_root", type=str, default=DEFAULT_IMAGES_ROOT)
    parser.add_argument("--edges_root", type=str, default=DEFAULT_EDGES_ROOT, help="Required for 8ch models.")
    parser.add_argument("--annotation_json", type=str, default=DEFAULT_ANNOTATION_JSON)
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)

    # Model specs
    parser.add_argument("--weight", type=str, default="")
    parser.add_argument("--weights", type=str, nargs="*", default=[])
    parser.add_argument("--default_unet", type=str, default=DEFAULT_UNET_NAME)

    # Device
    parser.add_argument("--gpus", type=str, default=_DEFAULT_GPUS)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")

    # Reproducibility
    parser.add_argument("--deterministic", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=str, default=SEEDS)

    # Optional subset
    parser.add_argument("--max_eval_images", type=int, default=0, help="0 => all annotated images.")

    # Ranking semantics
    parser.add_argument("--rank1_is_closest", type=str2bool, default=True)
    parser.add_argument("--pred_higher_is_farther", type=str2bool, default=True)
    parser.add_argument("--tie_epsilon", type=float, default=0.0)

    # Pred caching & saving
    parser.add_argument("--save_pred_depth", type=str2bool, default=True)
    parser.add_argument("--save_pred_vis", type=str2bool, default=True)
    parser.add_argument("--save_pred_rgb", type=str2bool, default=True, help="Only applies if out_channels=4 (RGBD).")
    parser.add_argument("--depth_vis_cmap", type=str, default="magma")

    parser.add_argument("--reuse_pred_depth", type=str2bool, default=True)
    parser.add_argument("--overwrite_pred_depth", type=str2bool, default=False)
    parser.add_argument(
        "--overwrite_missing_outputs",
        type=str2bool,
        default=False,
        help="If True and depth exists but rgb/vis is missing, regenerate by resampling.",
    )

    # Output control
    parser.add_argument("--flush_csv_each_seed", type=str2bool, default=True)
    parser.add_argument("--make_plots", type=str2bool, default=True)
    parser.add_argument("--make_summary", type=str2bool, default=True)
    parser.add_argument("--plots_format", type=str, default="png", help="Comma-separated: png,pdf,svg")

    # Custom labels
    parser.add_argument(
        "--label",
        type=str,
        nargs="*",
        default=[],
        help='Custom plot label rules: --label "match_substring=Nice Label" (repeatable).',
    )

    # UNet/diffusion core args (kept compatible; per-weight overrides apply)
    parser.add_argument("--input_size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--attention_resolutions", type=str, default=DEFAULT_ATTENTION_RESOLUTIONS)
    parser.add_argument("--in_channels", type=int, default=DEFAULT_IN_CHANNELS)
    parser.add_argument("--out_channels", type=int, default=4)

    parser.add_argument("--timesteps", type=int, default=250)
    parser.add_argument("--loss_type", type=str, default="l1", choices=["l1", "l2"])

    parser.add_argument("--num_channels", type=int, default=64)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_scale_shift_norm", type=str2bool, default=False)

    parser.add_argument("--use_ema", type=str2bool, default=True)

    args = parser.parse_args()

    args.attention_resolutions = normalize_attention_resolutions(args.attention_resolutions) or str(DEFAULT_ATTENTION_RESOLUTIONS)

    if str(args.default_unet).strip() not in UNET_NAME_TO_MODULE:
        known = ", ".join(sorted(UNET_NAME_TO_MODULE.keys()))
        raise ValueError(f"--default_unet must be one of: {known}")

    if int(args.in_channels) not in (2, 4, 5, 7, 8):
        raise ValueError("--in_channels must be one of 2,4,5,7,8.")

    if int(args.max_eval_images) < 0:
        raise ValueError("--max_eval_images must be >= 0.")

    if str(args.depth_vis_cmap).strip() == "":
        args.depth_vis_cmap = "magma"

    return args


def main():
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()