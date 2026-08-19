#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bench_depth_ranking_kvasirseg_from_preds.py

Compute depth-ranking metrics from already-computed depth PNGs (no model inference).

Metrics:
  - point_rank_accuracy
  - correct_count_out_of_5
  - exact_match
  - pairwise_accuracy
  - kendall_tau
  - spearman_rho

Key feature for mixed-resolution models (for example PPSNet 384x384):
  - Default evaluation uses each model's native depth resolution
  - GT points are mapped to the actual predicted depth size per image

Outputs under --output_root:
  - ranking_metrics.csv
  - ranking_per_image.csv
  - ranking_summary.csv
  - ranking_summary_clean_6sf.csv
  - plots/*.png (and/or pdf/svg)
  - plots/all_metric_bars_grid_1row.png
  - plots/paper_combined_metrics_heatmap.png
  - plots/paper_combined_metrics_heatmap__<cmap>.png (5 versions)
  - plots/paper_metrics_row_2plots_all5.png
  - plots/paper_metrics_row_3plots_all5.png
"""

import os
import re
import csv
import json
import math
import time
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ----------------------------
# Defaults
# ----------------------------
DEFAULT_ANNOTATION_JSON = "/well/rittscher/users/fxh757/Code/Kvasir_Ranking/Merged_PASS/merged_pass_consensus_gt.json"
DEFAULT_IMAGES_ROOT = "/well/rittscher/users/fxh757/Datasets/Kvasir-SEG/test/images/"

CHANNEL_COLORS = {
    "2ch": "#1f77b4",  # blue
    "4ch": "#ff7f0e",  # orange
    "5ch": "#9467bd",  # purple
    "7ch": "#17becf",  # cyan
    "8ch": "#8c564b",  # brown
    "Depth Anything V2": "#e377c2",  # pink (example)
}
PPSNET_COLOR = "#2ca02c"  # unique green

HEATMAP_CMAPS = ["viridis", "cividis", "magma", "plasma", "YlGnBu"]

DEFAULT_MODEL_DIRS = {
    # NoPhotoAug
    "2ch": "/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/final_ranking_eval_kvasirseg_paper/channel_study/unet_512_v2__ch2__sz256__attn64_32_16_8_4_2_1__results_2ch_256_2nd_noPhotoAug_307__model-52.pt/seed_0/pred_depth_u16",
    "4ch": "/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/final_ranking_eval_kvasirseg_paper/channel_study/unet_512_v2__ch4__sz256__attn64_32_16_8_4_2_1__results_4ch_256_2nd_noPhotoAug_307__model-104.pt/seed_0/pred_depth_u16",
    "5ch": "/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/final_ranking_eval_kvasirseg_paper/channel_study/unet_512_v2__ch5__sz256__attn64_32_16_8_4_2_1__results_5ch_256_2nd_noPhotoAug_307__model-104.pt/seed_0/pred_depth_u16/",
    "7ch": "/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/final_ranking_eval_kvasirseg_paper/channel_study/unet_512_v2__ch7__sz256__attn64_32_16_8_4_2_1__results_7ch_256NoPhotoAug_3nd_307__model-77.pt/seed_0/pred_depth_u16",
    "8ch": "/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/final_ranking_eval_kvasirseg_paper/channel_study/unet_512_v2__ch8__sz256__attn64_32_16_8_4_2_1__results_8ch_256_2nd_noPhotoAug_307__model-102.pt/seed_0/pred_depth_u16",

    # PhotoAug
    "2ch_photo": "/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/final_ranking_eval_kvasirseg_paper/channel_study/unet_512_v2__ch2__sz256__attn64_32_16_8_4_2_1__results_2ch_256_2nd_307__model-116.pt/seed_0/pred_depth_u16/",
    "4ch_photo": "/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/final_ranking_eval_kvasirseg_paper/channel_study/unet_512_v2__ch4__sz256__attn64_32_16_8_4_2_1__results_4ch_256_2nd_307__model-102.pt/seed_0/pred_depth_u16/",
    "5ch_photo": "/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/final_ranking_eval_kvasirseg_paper/channel_study/unet_512_v2__ch5__sz256__attn64_32_16_8_4_2_1__results_5ch_256_2nd_307__model-123.pt/seed_0/pred_depth_u16/",
    "7ch_photo": "/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/final_ranking_eval_kvasirseg_paper/channel_study/unet_512_v2__ch7__sz256__attn64_32_16_8_4_2_1__results_7ch_256_2nd_307__model-50.pt/seed_0/pred_depth_u16/",
    "8ch_photo": "/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/final_ranking_eval_kvasirseg_paper/channel_study/unet_512_v2__ch8__sz256__attn64_32_16_8_4_2_1__results_8ch_256_2nd_307__model-121.pt/seed_0/pred_depth_u16/",

    # PPSNet (native 384x384)
    "PPSNet": "/well/rittscher/users/fxh757/Code/PPSNet/logs_depth_ppsnet_kvasir/teacher/results_16bit/root",
    
    # Depth Anything
    "Depth Anything V2": "/well/rittscher/users/fxh757/Code/polyp-ddpm_polyp/split_2_zero/models_bench/DAV2/kva_depth",
}

DEFAULT_MODEL_ORDER = [
    "2ch", "2ch_photo",
    "4ch", "4ch_photo",
    "5ch", "5ch_photo",
    "7ch", "7ch_photo",
    "8ch", "8ch_photo",
    "PPSNet",
    "Depth Anything V2",
]
MODEL_ORDER_INDEX = {name: i for i, name in enumerate(DEFAULT_MODEL_ORDER)}

DEFAULT_INPUT_SIZE = 256

METRIC_SPECS = [
    {
        "key": "mean_point_rank_accuracy_mean",
        "std_key": "mean_point_rank_accuracy_std",
        "short": "Point rank acc",
        "title": "Mean point rank accuracy",
        "ylabel": "Accuracy",
        "fixed_bounds": (0.0, 1.0),
        "force_zero_based": True,
        "hide_negative_if_all_nonnegative": False,
        "line_color": "#1f77b4",
    },
    {
        "key": "pairwise_accuracy_mean",
        "std_key": "pairwise_accuracy_std",
        "short": "Pairwise acc",
        "title": "Pairwise ordering accuracy",
        "ylabel": "Accuracy",
        "fixed_bounds": (0.0, 1.0),
        "force_zero_based": True,
        "hide_negative_if_all_nonnegative": False,
        "line_color": "#ff7f0e",
    },
    {
        "key": "exact_match_rate_mean",
        "std_key": "exact_match_rate_std",
        "short": "Exact match",
        "title": "Exact match rate",
        "ylabel": "Rate",
        "fixed_bounds": (0.0, 1.0),
        "force_zero_based": True,
        "hide_negative_if_all_nonnegative": False,
        "line_color": "#2ca02c",
    },
    {
        "key": "kendall_tau_mean",
        "std_key": "kendall_tau_std",
        "short": "Kendall tau",
        "title": "Kendall tau",
        "ylabel": "Tau",
        "fixed_bounds": (-1.0, 1.0),
        "force_zero_based": False,
        "hide_negative_if_all_nonnegative": True,
        "line_color": "#9467bd",
    },
    {
        "key": "spearman_rho_mean",
        "std_key": "spearman_rho_std",
        "short": "Spearman rho",
        "title": "Spearman rho",
        "ylabel": "Rho",
        "fixed_bounds": (-1.0, 1.0),
        "force_zero_based": False,
        "hide_negative_if_all_nonnegative": True,
        "line_color": "#d62728",
    },
]


# =============================================================================
# DATA CLASSES
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


# =============================================================================
# GLOBAL WARN-ONCE
# =============================================================================
_WARNED_ONCE: set = set()


def _warn_once(key: str, msg: str):
    if key in _WARNED_ONCE:
        return
    _WARNED_ONCE.add(key)
    print(msg)


# =============================================================================
# UTILS
# =============================================================================
def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false).")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


def normalize_path(p: str) -> str:
    return os.path.abspath(os.path.expanduser(str(p)))


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


def _fmt_sig6(value: Any) -> str:
    """Format numeric values to 6 significant figures for clean CSVs."""
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        v = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(v):
        return ""
    return f"{v:.6g}"


# =============================================================================
# ANNOTATION LOADING
# =============================================================================
def load_annotation_json(annotation_path: str, images_root: str, max_images: int = 0) -> List[AnnImage]:
    annotation_path = normalize_path(annotation_path)
    images_root = normalize_path(images_root)

    if not os.path.isfile(annotation_path):
        raise FileNotFoundError(f"annotation_json not found: {annotation_path}")
    if not os.path.isdir(images_root):
        raise FileNotFoundError(f"images_root not found: {images_root}")

    with open(annotation_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "images" not in data:
        raise ValueError("Annotation JSON must be a dict with an 'images' key.")

    if "numPoints" in data:
        try:
            np_expected = int(data["numPoints"])
            if np_expected != 5:
                _warn_once("json_numPoints", f"[WARN] JSON header numPoints={np_expected} (expected 5 for this benchmark).")
        except Exception:
            pass

    ann_images: List[AnnImage] = []
    for rec in data.get("images", []):
        try:
            fn = str(rec["filename"])
            w = int(rec.get("width", 0) or 0)
            h = int(rec.get("height", 0) or 0)
            pts_raw = rec["points"]
        except Exception as e:
            _warn_once(f"bad_img_rec::{id(rec)}", f"[WARN] Skipping malformed image record: {e}")
            continue

        img_path = os.path.join(images_root, fn)
        if not os.path.isfile(img_path):
            _warn_once(f"missing_img::{fn}", f"[WARN] Image file referenced by annotation not found, skipping: {img_path}")
            continue

        try:
            with Image.open(img_path) as im:
                aw, ah = im.size
            if aw > 0 and ah > 0:
                if (w and h) and (aw != w or ah != h):
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

        ranks = [pp.rank for pp in pts]
        if any((r < 1 or r > 5) for r in ranks):
            _warn_once(f"rank_range::{fn}", f"[WARN] {fn}: some GT ranks are outside [1..5].")

        ann_images.append(AnnImage(filename=fn, path=img_path, width=w, height=h, points=pts))

        if max_images and len(ann_images) >= int(max_images):
            break

    if not ann_images:
        raise RuntimeError("No usable annotated images found (after filtering missing files).")

    return ann_images


# =============================================================================
# COORDINATE MAPPING
# =============================================================================
def map_point_to_resized(x: float, y: float, w: int, h: int, out_w: int, out_h: int) -> Tuple[int, int]:
    """
    Map original image coords to resized grid coords.

    x' = round(x / (W-1) * (out_w-1))
    y' = round(y / (H-1) * (out_h-1))
    """
    W = max(1, int(w))
    H = max(1, int(h))
    OW = max(1, int(out_w))
    OH = max(1, int(out_h))

    if W == 1:
        xr = 0
    else:
        xr = int(round(float(x) / float(W - 1) * float(OW - 1)))

    if H == 1:
        yr = 0
    else:
        yr = int(round(float(y) / float(H - 1) * float(OH - 1)))

    xr = clamp_int(xr, 0, OW - 1)
    yr = clamp_int(yr, 0, OH - 1)
    return xr, yr


# =============================================================================
# DEPTH LOADING
# =============================================================================
def load_depth_png_as_01(depth_path: str, input_size: Optional[int] = None) -> np.ndarray:
    """
    Load predicted depth PNG -> float32 depth in [0, 1].

    Supports:
      - 8-bit grayscale
      - 16-bit grayscale
      - multi-channel PNG (uses channel 0 and warns once)

    If input_size is None:
      - keep native PNG shape

    If input_size is int:
      - resize to (input_size, input_size) with nearest-neighbor
    """
    im = Image.open(depth_path)
    arr = np.array(im)

    if arr.ndim == 3:
        _warn_once(f"rgb_depth::{depth_path}", f"[WARN] Depth PNG appears multi-channel, using first channel: {depth_path}")
        arr = arr[..., 0]

    arr = arr.astype(np.float32)
    if float(arr.max()) <= 255.0 + 1e-6:
        d01 = arr / 255.0
    else:
        d01 = arr / 65535.0
    d01 = np.clip(d01, 0.0, 1.0)

    if input_size is not None:
        s = int(input_size)
        if d01.shape[0] != s or d01.shape[1] != s:
            im16 = Image.fromarray((d01 * 65535.0 + 0.5).astype(np.uint16), mode="I;16")
            im16 = im16.resize((s, s), resample=Image.NEAREST)
            d01 = np.array(im16).astype(np.float32) / 65535.0
            d01 = np.clip(d01, 0.0, 1.0)

    return d01.astype(np.float32)


# =============================================================================
# RANKING METRICS
# =============================================================================
def _spearman_rho_from_lists(gt_ranks: List[int], pred_vals: List[float]) -> float:
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

    x_c = x - float(x.mean())
    y_c = y - float(y.mean())
    denom = float(np.sqrt((x_c * x_c).sum() * (y_c * y_c).sum()))
    if denom <= 1e-12:
        return float("nan")
    return float((x_c * y_c).sum() / denom)


def compute_ranking_metrics_for_image(
    *,
    gt_points: List[AnnPoint],
    pred01_map: np.ndarray,
    orig_w: int,
    orig_h: int,
    rank1_is_closest: bool,
    pred_higher_is_farther: bool,
    tie_epsilon: float,
) -> Tuple[PerImageRankingMetrics, Dict[int, float], Dict[int, int], Dict[int, int], Dict[int, Tuple[int, int]]]:
    if pred01_map.ndim != 2:
        raise ValueError(f"pred01_map must be 2D [H,W], got shape {pred01_map.shape}")

    pred_h = int(pred01_map.shape[0])
    pred_w = int(pred01_map.shape[1])

    if rank1_is_closest:
        key_sign = +1.0 if pred_higher_is_farther else -1.0
    else:
        key_sign = -1.0 if pred_higher_is_farther else +1.0

    pred_values: Dict[int, float] = {}
    gt_ranks: Dict[int, int] = {}
    mapped_xy: Dict[int, Tuple[int, int]] = {}

    for p in gt_points:
        xr, yr = map_point_to_resized(p.x, p.y, orig_w, orig_h, pred_w, pred_h)
        mapped_xy[p.point_id] = (xr, yr)
        pred_values[p.point_id] = float(pred01_map[yr, xr])
        gt_ranks[p.point_id] = int(p.rank)

    point_ids = [p.point_id for p in gt_points]
    point_ids_sorted = sorted(point_ids, key=lambda pid: (key_sign * pred_values[pid], pid))

    pred_ranks: Dict[int, int] = {}
    for i, pid in enumerate(point_ids_sorted):
        pred_ranks[pid] = i + 1

    total_points = len(point_ids)
    correct_points = sum(1 for pid in point_ids if pred_ranks.get(pid, -999) == gt_ranks.get(pid, -888))
    point_rank_accuracy = float(correct_points) / float(total_points) if total_points > 0 else float("nan")

    correct_count_out_of_5 = int(correct_points)
    exact_match = 1.0 if (correct_points == total_points and total_points > 0) else 0.0

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
# MODEL DIR RESOLUTION
# =============================================================================
_SEED_DIR_RE = re.compile(r"^seed_(\d+)$")


def _dir_has_pngs(d: str) -> bool:
    if not os.path.isdir(d):
        return False
    for fn in os.listdir(d):
        if fn.lower().endswith(".png"):
            return True
    return False


def resolve_depth_dirs_for_model(model_path: str) -> List[Tuple[int, str, str]]:
    """
    Return list of (seed_int, seed_label, depth_dir).

    Supported:
      - model_path/*.png
      - model_path/pred_depth_u16/*.png
      - model_path/seed_*/pred_depth_u16/*.png
      - model_path/seed_*/*.png
    """
    p = normalize_path(model_path)
    if not os.path.isdir(p):
        raise FileNotFoundError(f"Model path is not a directory: {p}")

    if _dir_has_pngs(p):
        return [(0, "seed_0", p)]

    pd = os.path.join(p, "pred_depth_u16")
    if _dir_has_pngs(pd):
        return [(0, "seed_0", pd)]

    items: List[Tuple[int, str, str]] = []
    for name in sorted(os.listdir(p)):
        m = _SEED_DIR_RE.match(name)
        if not m:
            continue
        seed_int = int(m.group(1))
        seed_dir = os.path.join(p, name)
        pd2 = os.path.join(seed_dir, "pred_depth_u16")
        if _dir_has_pngs(pd2):
            items.append((seed_int, name, pd2))
        elif _dir_has_pngs(seed_dir):
            items.append((seed_int, name, seed_dir))

    if items:
        return items

    raise FileNotFoundError(
        f"Could not find depth PNGs under '{p}'. Expected one of:\n"
        f"  - {p}/*.png\n"
        f"  - {p}/pred_depth_u16/*.png\n"
        f"  - {p}/seed_*/pred_depth_u16/*.png\n"
        f"  - {p}/seed_*/*.png"
    )


def parse_model_dirs_override(model_dir_args: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in model_dir_args or []:
        s = str(item).strip()
        if not s:
            continue
        if "=" not in s:
            raise ValueError(f"--model_dir expects 'NAME=PATH', got: {s}")
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k or not v:
            raise ValueError(f"--model_dir expects 'NAME=PATH', got: {s}")
        out[k] = v
    return out


# =============================================================================
# CSV FIELDS
# =============================================================================
MAIN_CSV_FIELDS = [
    "model_name",
    "model_path",
    "seed",
    "seed_label",
    "depth_dir",
    "images_root",
    "annotation_json",
    "eval_native_size",
    "input_size",
    "invert_depth",
    "n_images_total",
    "n_images_used",
    "n_images_missing",
    "point_acc_overall",
    "mean_point_rank_accuracy",
    "exact_match_rate",
    "pairwise_accuracy_mean",
    "kendall_tau_mean",
    "spearman_rho_mean",
    "n_img_correct_5",
    "n_img_correct_4",
    "n_img_correct_3",
    "n_img_correct_2",
    "n_img_correct_1",
    "n_img_correct_0",
]

PER_IMAGE_CSV_FIELDS = [
    "model_name",
    "model_path",
    "seed",
    "seed_label",
    "depth_dir",
    "images_root",
    "annotation_json",
    "eval_native_size",
    "input_size",
    "invert_depth",
    "filename",
    "pred_width",
    "pred_height",
    "point_rank_accuracy",
    "correct_count_out_of_5",
    "exact_match",
    "pairwise_accuracy",
    "kendall_tau",
    "spearman_rho",
    "pid_1", "pid_2", "pid_3", "pid_4", "pid_5",
    "gt_rank_1", "gt_rank_2", "gt_rank_3", "gt_rank_4", "gt_rank_5",
    "pred_rank_1", "pred_rank_2", "pred_rank_3", "pred_rank_4", "pred_rank_5",
    "pred_val_1", "pred_val_2", "pred_val_3", "pred_val_4", "pred_val_5",
    "mapped_x_1", "mapped_y_1",
    "mapped_x_2", "mapped_y_2",
    "mapped_x_3", "mapped_y_3",
    "mapped_x_4", "mapped_y_4",
    "mapped_x_5", "mapped_y_5",
]

SUMMARY_CSV_FIELDS = [
    "label",
    "model_name",
    "model_path",
    "n_seeds",
    "n_images_total",
    "n_images_used_mean",
    "n_images_used_std",
    "n_images_missing_mean",
    "n_images_missing_std",
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

# Clean summary CSV for paper/tables, no paths or bookkeeping columns
SUMMARY_CLEAN_6SF_FIELDS = [
    "model_name",
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


def write_clean_summary_csv_6sf(path: str, summary_rows: List[Dict[str, Any]]):
    clean_rows: List[Dict[str, str]] = []
    for r in summary_rows:
        row: Dict[str, str] = {"model_name": str(r.get("model_name", ""))}
        for k in SUMMARY_CLEAN_6SF_FIELDS:
            if k == "model_name":
                continue
            row[k] = _fmt_sig6(r.get(k, ""))
        clean_rows.append(row)
    write_csv(path, SUMMARY_CLEAN_6SF_FIELDS, clean_rows)
    print(f"[CSV] Wrote clean summary (6 sf): {path}")


# =============================================================================
# LABEL RULES
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


def label_for_model_name(model_name: str, model_path: str, label_rules: List[Tuple[str, str]]) -> str:
    hay = f"{model_name} | {model_path}"
    for match, label in label_rules:
        if match in hay:
            return label
    return str(model_name)


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
# PLOT HELPERS
# =============================================================================
def _apply_paper_rcparams():
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _base_channel_key(model_name: str) -> Optional[str]:
    for k in CHANNEL_COLORS.keys():
        if model_name == k or model_name.startswith(f"{k}_"):
            return k
    return None


def _is_photo_model(model_name: str) -> bool:
    return str(model_name).endswith("_photo")


def _bar_style_for_model(model_name: str) -> Dict[str, Any]:
    if str(model_name) == "PPSNet":
        return {
            "color": PPSNET_COLOR,
            "edgecolor": "#222222",
            "linewidth": 0.9,
            "hatch": "",
            "alpha": 0.95,
        }
    base = _base_channel_key(str(model_name))
    color = CHANNEL_COLORS.get(base, "#7f7f7f")
    if _is_photo_model(model_name):
        return {
            "color": color,
            "edgecolor": "#222222",
            "linewidth": 0.9,
            "hatch": "//",
            "alpha": 0.92,
        }
    return {
        "color": color,
        "edgecolor": "#222222",
        "linewidth": 0.9,
        "hatch": "",
        "alpha": 0.95,
    }


def _photo_hatch_legend_handles() -> List[mpatches.Patch]:
    noaug_patch = mpatches.Patch(facecolor="#DDDDDD", edgecolor="#222222", label="NoPhotoAug")
    photo_patch = mpatches.Patch(facecolor="#DDDDDD", edgecolor="#222222", hatch="//", label="PhotoAug")
    return [noaug_patch, photo_patch]


def _compute_smart_ylim(
    means: List[float],
    stds: List[float],
    *,
    fixed_bounds: Optional[Tuple[float, float]] = None,
    hide_negative_if_all_nonnegative: bool = False,
    force_zero_based: bool = False,
) -> Tuple[float, float]:
    lows = []
    highs = []
    for m, s in zip(means, stds):
        if not math.isfinite(m):
            continue
        e = s if (math.isfinite(s) and s >= 0.0) else 0.0
        lows.append(m - e)
        highs.append(m + e)

    if len(lows) == 0:
        return fixed_bounds if fixed_bounds is not None else (0.0, 1.0)

    lo = min(lows)
    hi = max(highs)

    if fixed_bounds is not None:
        lo = max(lo, fixed_bounds[0])
        hi = min(hi, fixed_bounds[1])

    if hide_negative_if_all_nonnegative and lo >= -1e-12:
        lo = 0.0
    if force_zero_based and lo > 0.0:
        lo = 0.0

    if hi < lo:
        hi = lo

    span = hi - lo
    pad = 0.08 * span if span >= 1e-9 else 0.05
    lo -= pad
    hi += pad

    if fixed_bounds is not None:
        lo = max(lo, fixed_bounds[0])
        hi = min(hi, fixed_bounds[1])

    if hide_negative_if_all_nonnegative and min(lows) >= -1e-12:
        lo = max(lo, 0.0)

    return float(lo), float(hi)


def _sort_rows_by_metric(summary_rows: List[Dict[str, Any]], metric_mean_key: str) -> List[Dict[str, Any]]:
    def _key(r: Dict[str, Any]):
        v = _safe_float(r.get(metric_mean_key, "nan"))
        if not math.isfinite(v):
            return (1, 1e9, str(r.get("label", "")))
        return (0, -v, str(r.get("label", "")))
    return sorted(summary_rows, key=_key)


def _sort_rows_by_exact_match(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _key(r: Dict[str, Any]):
        ex = _safe_float(r.get("exact_match_rate_mean", "nan"))
        mp = _safe_float(r.get("mean_point_rank_accuracy_mean", "nan"))
        pw = _safe_float(r.get("pairwise_accuracy_mean", "nan"))
        if not math.isfinite(ex):
            ex = -1e9
        if not math.isfinite(mp):
            mp = -1e9
        if not math.isfinite(pw):
            pw = -1e9
        return (0, -ex, -mp, -pw, str(r.get("label", "")))
    return sorted(summary_rows, key=_key)


def _sort_rows_by_correct_count_distribution(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def expected_correct(r: Dict[str, Any]) -> float:
        total = 0.0
        for k in range(0, 6):
            p = _safe_float(r.get(f"prop_img_correct_{k}_mean", 0.0))
            if math.isfinite(p):
                total += k * p
        return total

    def _key(r: Dict[str, Any]):
        score = expected_correct(r)
        if not math.isfinite(score):
            return (1, 1e9, str(r.get("label", "")))
        m = _safe_float(r.get("mean_point_rank_accuracy_mean", "nan"))
        if not math.isfinite(m):
            m = -1e9
        return (0, -score, -m, str(r.get("label", "")))
    return sorted(summary_rows, key=_key)


def _style_model_xticklabels(ax, model_names: List[str]):
    for tick, mn in zip(ax.get_xticklabels(), model_names):
        st = _bar_style_for_model(mn)
        tick.set_color(st["color"])
        if _is_photo_model(mn):
            tick.set_fontstyle("italic")


def _add_center_watermark(ax, letter: str):
    ax.text(
        0.5,
        0.5,
        letter,
        transform=ax.transAxes,
        fontsize=32,
        fontweight="bold",
        color="black",
        alpha=0.15,
        ha="center",
        va="center",
        zorder=0,
    )


def _color_value_for_heatmap(raw_value: float) -> float:
    """
    Use RAW metric value directly for heatmap color.
    Assumes metrics are already in [0, 1].
    Any out-of-range values are clipped for safety.
    """
    if not math.isfinite(raw_value):
        return float("nan")
    return float(np.clip(raw_value, 0.0, 1.0))


def _prepare_metric_matrices_exact_sorted(summary_rows: List[Dict[str, Any]]):
    rows_sorted = _sort_rows_by_exact_match(summary_rows)
    labels = [str(r["label"]) for r in rows_sorted]
    model_names = [str(r["model_name"]) for r in rows_sorted]

    n_metrics = len(METRIC_SPECS)
    n_models = len(rows_sorted)
    mat_raw = np.full((n_metrics, n_models), np.nan, dtype=np.float64)
    mat_color = np.full((n_metrics, n_models), np.nan, dtype=np.float64)

    for i, spec in enumerate(METRIC_SPECS):
        for j, r in enumerate(rows_sorted):
            rv = _safe_float(r.get(spec["key"], "nan"))
            mat_raw[i, j] = rv
            mat_color[i, j] = _color_value_for_heatmap(rv)

    mat_rank = np.full((n_metrics, n_models), np.nan, dtype=np.float64)
    for i in range(n_metrics):
        vals = mat_raw[i, :].copy()
        valid = np.isfinite(vals)
        if not np.any(valid):
            continue
        valid_idx = np.where(valid)[0]
        order = np.argsort(-vals[valid])  # larger is better
        ranks = np.empty(np.sum(valid), dtype=np.float64)
        ranks[order] = np.arange(1, np.sum(valid) + 1, dtype=np.float64)
        mat_rank[i, valid_idx] = ranks

    return rows_sorted, labels, model_names, mat_raw, mat_color, mat_rank


def _draw_photo_hatch_legend_inside(ax):
    handles = _photo_hatch_legend_handles()
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=True,
        framealpha=0.92,
        borderpad=0.35,
        handlelength=1.4,
    )


def _draw_metric_bars_on_ax(
    ax,
    rows_sorted: List[Dict[str, Any]],
    spec: Dict[str, Any],
    *,
    annotate: bool = True,
    add_hatch_legend: bool = False,
    watermark_letter: Optional[str] = None,
):
    labels = [str(r["label"]) for r in rows_sorted]
    model_names = [str(r["model_name"]) for r in rows_sorted]
    means = [_safe_float(r.get(spec["key"], "nan")) for r in rows_sorted]
    stds = [_safe_float(r.get(spec["std_key"], "nan")) for r in rows_sorted]
    x = np.arange(len(labels), dtype=np.float32)

    if watermark_letter:
        _add_center_watermark(ax, watermark_letter)

    for i, (m, s, mn) in enumerate(zip(means, stds, model_names)):
        if not math.isfinite(m):
            continue
        e = s if (math.isfinite(s) and s >= 0.0) else 0.0
        st = _bar_style_for_model(mn)
        bars = ax.bar(
            [x[i]],
            [m],
            yerr=[e],
            capsize=3.0,
            width=0.82,
            color=st["color"],
            edgecolor=st["edgecolor"],
            linewidth=st["linewidth"],
            alpha=st["alpha"],
            zorder=3,
        )
        bars.patches[0].set_hatch(st["hatch"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    _style_model_xticklabels(ax, model_names)
    ax.set_title(f"{spec['title']} (mean ± std)")
    ax.set_ylabel(spec["ylabel"])
    ax.grid(True, axis="y", linestyle="--", alpha=0.25, zorder=0)

    ylim = _compute_smart_ylim(
        means,
        stds,
        fixed_bounds=spec["fixed_bounds"],
        hide_negative_if_all_nonnegative=bool(spec["hide_negative_if_all_nonnegative"]),
        force_zero_based=bool(spec["force_zero_based"]),
    )
    ax.set_ylim(*ylim)

    if annotate:
        y_span = max(1e-6, ylim[1] - ylim[0])
        for i, (m, s) in enumerate(zip(means, stds)):
            if not math.isfinite(m):
                continue
            e = s if (math.isfinite(s) and s >= 0.0) else 0.0
            y_text = min(m + e + 0.012 * y_span, ylim[1] - 0.02 * y_span)
            ax.text(i, y_text, f"{m:.3f}", ha="center", va="bottom", fontsize=7.4, zorder=4)

    if add_hatch_legend:
        _draw_photo_hatch_legend_inside(ax)


def _save_individual_metric_plot(summary_rows: List[Dict[str, Any]], spec: Dict[str, Any], out_base_no_ext: str, formats: List[str]):
    rows_sorted = _sort_rows_by_metric(summary_rows, spec["key"])
    fig = plt.figure(figsize=(max(12, 0.72 * len(rows_sorted)), 6.2))
    ax = fig.add_subplot(1, 1, 1)
    _draw_metric_bars_on_ax(ax, rows_sorted, spec, annotate=True, add_hatch_legend=True)
    fig.tight_layout()
    for fmt in formats:
        out_path = f"{out_base_no_ext}.{fmt}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[PLOT] Saved: {out_path}")
    plt.close(fig)


def _plot_stacked_distribution(summary_rows: List[Dict[str, Any]], out_base_no_ext: str, formats: List[str]):
    rows_sorted = _sort_rows_by_correct_count_distribution(summary_rows)
    labels = [str(r["label"]) for r in rows_sorted]
    model_names = [str(r["model_name"]) for r in rows_sorted]
    x = np.arange(len(labels), dtype=np.float32)
    styles = [_bar_style_for_model(mn) for mn in model_names]

    stack_colors = {
        0: "#d73027",
        1: "#fc8d59",
        2: "#fee08b",
        3: "#d9ef8b",
        4: "#91cf60",
        5: "#1a9850",
    }

    fig = plt.figure(figsize=(max(12, 0.78 * len(labels)), 6.5))
    ax = fig.add_subplot(1, 1, 1)

    bottom = np.zeros(len(labels), dtype=np.float32)
    for k in range(0, 6):
        vals = np.array([_safe_float(r.get(f"prop_img_correct_{k}_mean", 0.0)) for r in rows_sorted], dtype=np.float32)
        bars = ax.bar(
            x,
            vals,
            bottom=bottom,
            label=f"{k}/5",
            width=0.82,
            color=stack_colors[k],
            edgecolor=[st["edgecolor"] for st in styles],
            linewidth=0.7,
            alpha=0.95,
            zorder=3,
        )
        for patch, st in zip(bars.patches, styles):
            patch.set_hatch(st["hatch"])
        bottom += vals

    for i, r in enumerate(rows_sorted):
        expected_k = 0.0
        for k in range(0, 6):
            p = _safe_float(r.get(f"prop_img_correct_{k}_mean", 0.0))
            if math.isfinite(p):
                expected_k += k * p
        ax.text(i, 1.01, f"{expected_k:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    _style_model_xticklabels(ax, model_names)
    ax.set_title("Correct ranks per image (out of 5), mean proportion across seeds")
    ax.set_ylabel("Proportion of images")
    ax.set_ylim(0.0, 1.08)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25, zorder=0)

    leg1 = ax.legend(
        title="Correct points",
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.985),
        frameon=True,
        framealpha=0.92,
        borderpad=0.35,
    )
    ax.add_artist(leg1)
    _draw_photo_hatch_legend_inside(ax)

    fig.tight_layout()
    for fmt in formats:
        out_path = f"{out_base_no_ext}.{fmt}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[PLOT] Saved: {out_path}")
    plt.close(fig)


def _plot_all_metric_bars_grid_1row(summary_rows: List[Dict[str, Any]], out_png_path: str):
    letters = ["A", "B", "C", "D", "E"]
    fig = plt.figure(figsize=(32, 5.6))
    axes = [fig.add_subplot(1, 5, i + 1) for i in range(5)]

    for i, (ax, spec) in enumerate(zip(axes, METRIC_SPECS)):
        rows_sorted = _sort_rows_by_metric(summary_rows, spec["key"])
        _draw_metric_bars_on_ax(
            ax=ax,
            rows_sorted=rows_sorted,
            spec=spec,
            annotate=False,
            add_hatch_legend=(i == 4),
            watermark_letter=letters[i],
        )

    fig.suptitle("Ranking metrics by model (each panel sorted left to right by that metric)", y=1.02, fontsize=12)
    fig.tight_layout()
    ensure_dir(os.path.dirname(out_png_path))
    fig.savefig(out_png_path, dpi=300, bbox_inches="tight")
    print(f"[PLOT] Saved: {out_png_path}")
    plt.close(fig)


def _draw_heatmap_panel(
    ax,
    labels: List[str],
    model_names: List[str],
    mat_raw: np.ndarray,
    mat_color: np.ndarray,
    *,
    title: str,
    cmap_name: str = "viridis",
    add_panel_letter: Optional[str] = None,
):
    if add_panel_letter:
        _add_center_watermark(ax, add_panel_letter)

    # Raw values directly drive the colors (expected range 0..1)
    im = ax.imshow(mat_color, aspect="auto", cmap=cmap_name, vmin=0.0, vmax=1.0, zorder=2)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    _style_model_xticklabels(ax, model_names)
    ax.set_yticks(np.arange(len(METRIC_SPECS)))
    ax.set_yticklabels([s["short"] for s in METRIC_SPECS])
    ax.set_title(title)

    for i in range(mat_raw.shape[0]):
        for j in range(mat_raw.shape[1]):
            rv = mat_raw[i, j]
            cv = mat_color[i, j]
            txt = "NA" if not np.isfinite(rv) else f"{rv:.3f}"
            txt_color = "white" if (np.isfinite(cv) and cv < 0.45) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=txt_color, zorder=3)

    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(METRIC_SPECS), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6, alpha=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def _draw_rank_heatmap_panel(
    ax,
    labels: List[str],
    model_names: List[str],
    mat_rank: np.ndarray,
    *,
    title: str,
    cmap_name: str = "magma_r",
    add_panel_letter: Optional[str] = None,
):
    if add_panel_letter:
        _add_center_watermark(ax, add_panel_letter)

    valid = np.isfinite(mat_rank)
    vmax = int(np.nanmax(mat_rank)) if np.any(valid) else 1
    im = ax.imshow(mat_rank, aspect="auto", cmap=cmap_name, vmin=1, vmax=max(1, vmax), zorder=2)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    _style_model_xticklabels(ax, model_names)
    ax.set_yticks(np.arange(len(METRIC_SPECS)))
    ax.set_yticklabels([s["short"] for s in METRIC_SPECS])
    ax.set_title(title)

    for i in range(mat_rank.shape[0]):
        for j in range(mat_rank.shape[1]):
            rv = mat_rank[i, j]
            txt = "NA" if not np.isfinite(rv) else f"{int(rv)}"
            txt_color = "white" if (np.isfinite(rv) and rv > (0.55 * vmax)) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=txt_color, zorder=3)

    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(METRIC_SPECS), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6, alpha=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def _draw_metric_profile_panel(
    ax,
    rows_sorted: List[Dict[str, Any]],
    labels: List[str],
    model_names: List[str],
    *,
    title: str,
    add_metric_legend: bool = True,
    add_panel_letter: Optional[str] = None,
):
    if add_panel_letter:
        _add_center_watermark(ax, add_panel_letter)

    x = np.arange(len(labels), dtype=np.float64)

    for spec in METRIC_SPECS:
        y = []
        for r in rows_sorted:
            rv = _safe_float(r.get(spec["key"], "nan"))
            y.append(_color_value_for_heatmap(rv))
        y = np.array(y, dtype=np.float64)

        ax.plot(
            x,
            y,
            marker="o",
            markersize=3.5,
            linewidth=1.7,
            label=spec["short"],
            color=spec["line_color"],
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    _style_model_xticklabels(ax, model_names)
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25, zorder=0)

    if add_metric_legend:
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(0.985, 0.985),
            frameon=True,
            framealpha=0.92,
            borderpad=0.35,
        )


def _plot_paper_combined_metrics_heatmap_variants(summary_rows: List[Dict[str, Any]], out_dir: str):
    """
    Save 5 versions of the combined heatmap in different colormaps.
    Also saves a default alias: paper_combined_metrics_heatmap.png (first cmap).
    """
    rows_sorted, labels, model_names, mat_raw, mat_color, _ = _prepare_metric_matrices_exact_sorted(summary_rows)
    _ = rows_sorted
    ensure_dir(out_dir)

    for idx, cmap_name in enumerate(HEATMAP_CMAPS):
        fig_w = max(12, 0.58 * len(labels) + 3.0)
        fig = plt.figure(figsize=(fig_w, 4.9), constrained_layout=True)
        gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[40, 1.2])

        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])

        im = _draw_heatmap_panel(
            ax=ax,
            labels=labels,
            model_names=model_names,
            mat_raw=mat_raw,
            mat_color=mat_color,
            title="Combined ranking metrics heatmap (sorted by exact match)",
            cmap_name=cmap_name,
            add_panel_letter="A",
        )
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Metric value")

        fig.text(
            0.01, 0.01,
            "X label colors show channel family. Italic x labels indicate PhotoAug.",
            fontsize=7.5,
        )

        out_cmap = os.path.join(out_dir, f"paper_combined_metrics_heatmap__{cmap_name}.png")
        fig.savefig(out_cmap, dpi=300, bbox_inches="tight")
        print(f"[PLOT] Saved: {out_cmap}")

        if idx == 0:
            out_alias = os.path.join(out_dir, "paper_combined_metrics_heatmap.png")
            fig.savefig(out_alias, dpi=300, bbox_inches="tight")
            print(f"[PLOT] Saved: {out_alias}")

        plt.close(fig)


def _plot_paper_metrics_row_2plots_all5(summary_rows: List[Dict[str, Any]], out_png_path: str):
    """
    2-panel compact figure, still contains all 5 metrics:
      A: raw-values heatmap (raw colors from metric values)
      B: raw metric profile lines
    """
    rows_sorted, labels, model_names, mat_raw, mat_color, _ = _prepare_metric_matrices_exact_sorted(summary_rows)

    fig = plt.figure(figsize=(22, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[1.20, 0.055, 1.45])

    ax1 = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    im = _draw_heatmap_panel(
        ax=ax1,
        labels=labels,
        model_names=model_names,
        mat_raw=mat_raw,
        mat_color=mat_color,
        title="A  Raw values heatmap (sorted by exact match)",
        cmap_name="viridis",
        add_panel_letter="A",
    )
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Metric value")

    _draw_metric_profile_panel(
        ax=ax2,
        rows_sorted=rows_sorted,
        labels=labels,
        model_names=model_names,
        title="B  Metric profile across models",
        add_metric_legend=True,
        add_panel_letter="B",
    )

    fig.suptitle("Compact paper view (2 panels) with all five metrics", y=1.02, fontsize=12)
    ensure_dir(os.path.dirname(out_png_path))
    fig.savefig(out_png_path, dpi=300, bbox_inches="tight")
    print(f"[PLOT] Saved: {out_png_path}")
    plt.close(fig)


def _plot_paper_metrics_row_3plots_all5(summary_rows: List[Dict[str, Any]], out_png_path: str):
    """
    3-panel compact figure, still contains all 5 metrics:
      A: raw-values heatmap
      B: raw metric profile lines
      C: per-metric rank heatmap
    """
    rows_sorted, labels, model_names, mat_raw, mat_color, mat_rank = _prepare_metric_matrices_exact_sorted(summary_rows)

    fig = plt.figure(figsize=(30, 5.9), constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=1,
        ncols=5,
        width_ratios=[1.18, 0.05, 1.12, 1.18, 0.05]
    )

    ax1 = fig.add_subplot(gs[0, 0])
    cax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 3])
    cax3 = fig.add_subplot(gs[0, 4])

    im1 = _draw_heatmap_panel(
        ax=ax1,
        labels=labels,
        model_names=model_names,
        mat_raw=mat_raw,
        mat_color=mat_color,
        title="A  Raw values heatmap (sorted by exact match)",
        cmap_name="viridis",
        add_panel_letter="A",
    )
    cb1 = fig.colorbar(im1, cax=cax1)
    cb1.set_label("Metric value")

    _draw_metric_profile_panel(
        ax=ax2,
        rows_sorted=rows_sorted,
        labels=labels,
        model_names=model_names,
        title="B  Metric profile",
        add_metric_legend=True,
        add_panel_letter="B",
    )

    im3 = _draw_rank_heatmap_panel(
        ax=ax3,
        labels=labels,
        model_names=model_names,
        mat_rank=mat_rank,
        title="C  Per-metric rank heatmap (1 is best)",
        cmap_name="magma_r",
        add_panel_letter="C",
    )
    cb3 = fig.colorbar(im3, cax=cax3)
    cb3.set_label("Rank")

    fig.suptitle("Compact paper view (3 panels) with all five metrics", y=1.02, fontsize=12)
    ensure_dir(os.path.dirname(out_png_path))
    fig.savefig(out_png_path, dpi=300, bbox_inches="tight")
    print(f"[PLOT] Saved: {out_png_path}")
    plt.close(fig)


# =============================================================================
# CORE EVAL
# =============================================================================
def eval_one_model_one_seed_from_depth_dir(
    *,
    model_name: str,
    model_path: str,
    seed_int: int,
    seed_label: str,
    depth_dir: str,
    ann_images: List[AnnImage],
    eval_native_size: bool,
    input_size: int,
    rank1_is_closest: bool,
    pred_higher_is_farther: bool,
    tie_epsilon: float,
    strict_missing: bool,
    invert_depth: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    depth_dir = normalize_path(depth_dir)
    if not os.path.isdir(depth_dir):
        raise FileNotFoundError(f"depth_dir not found: {depth_dir}")

    total_correct_points = 0
    total_points = 0

    sum_point_rank_acc = 0.0
    sum_exact = 0.0
    sum_pairwise = 0.0
    sum_tau = 0.0
    sum_rho = 0.0

    img_correct_count_bins = {k: 0 for k in range(0, 6)}
    per_image_rows: List[Dict[str, Any]] = []

    n_missing = 0
    n_used = 0

    for ai in ann_images:
        stem = os.path.splitext(os.path.basename(ai.filename))[0]
        depth_path = os.path.join(depth_dir, f"{stem}.png")

        if not os.path.isfile(depth_path):
            n_missing += 1
            msg = f"[MISSING] model={model_name} {seed_label} depth not found: {depth_path}"
            if strict_missing:
                raise FileNotFoundError(msg)
            _warn_once(f"missing_depth::{model_name}::{seed_label}::{stem}", msg)
            continue

        if eval_native_size:
            pred01 = load_depth_png_as_01(depth_path, input_size=None)
        else:
            pred01 = load_depth_png_as_01(depth_path, input_size=int(input_size))

        if bool(invert_depth):
            pred01 = 1.0 - pred01

        pred_h = int(pred01.shape[0])
        pred_w = int(pred01.shape[1])

        metrics, pred_vals, pred_ranks, gt_ranks, mapped_xy = compute_ranking_metrics_for_image(
            gt_points=ai.points,
            pred01_map=pred01,
            orig_w=ai.width,
            orig_h=ai.height,
            rank1_is_closest=bool(args_rank1_is_closest := rank1_is_closest),
            pred_higher_is_farther=bool(args_pred_higher_is_farther := pred_higher_is_farther),
            tie_epsilon=float(tie_epsilon),
        )
        _ = args_rank1_is_closest
        _ = args_pred_higher_is_farther

        n_used += 1
        total_correct_points += int(metrics.correct_points)
        total_points += int(metrics.total_points)

        sum_point_rank_acc += float(metrics.point_rank_accuracy)
        sum_exact += float(metrics.exact_match)
        sum_pairwise += float(metrics.pairwise_accuracy)
        sum_tau += float(metrics.kendall_tau)
        sum_rho += float(metrics.spearman_rho)

        cc = clamp_int(int(metrics.correct_count_out_of_5), 0, 5)
        img_correct_count_bins[cc] += 1

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
            model_name=str(model_name),
            model_path=str(model_path),
            seed=int(seed_int),
            seed_label=str(seed_label),
            depth_dir=str(depth_dir),
            images_root="",
            annotation_json="",
            eval_native_size=bool(eval_native_size),
            input_size=int(input_size),
            invert_depth=bool(invert_depth),
            filename=str(ai.filename),
            pred_width=int(pred_w),
            pred_height=int(pred_h),
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

    n_total = len(ann_images)
    if n_used <= 0:
        raise RuntimeError(f"No images evaluated for model={model_name} {seed_label}. Missing={n_missing}/{n_total}")

    point_acc_overall = float(total_correct_points) / float(total_points) if total_points > 0 else float("nan")
    mean_point_rank_accuracy = float(sum_point_rank_acc) / float(n_used)
    exact_match_rate = float(sum_exact) / float(n_used)
    pairwise_accuracy_mean = float(sum_pairwise) / float(n_used)
    kendall_tau_mean = float(sum_tau) / float(n_used)
    spearman_rho_mean = float(sum_rho) / float(n_used)

    main_row = dict(
        model_name=str(model_name),
        model_path=str(model_path),
        seed=int(seed_int),
        seed_label=str(seed_label),
        depth_dir=str(depth_dir),
        images_root="",
        annotation_json="",
        eval_native_size=bool(eval_native_size),
        input_size=int(input_size),
        invert_depth=bool(invert_depth),
        n_images_total=int(n_total),
        n_images_used=int(n_used),
        n_images_missing=int(n_missing),
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
# SUMMARY + PLOTS
# =============================================================================
def generate_summary_and_plots(
    *,
    main_rows: List[Dict[str, Any]],
    summary_csv_path: str,
    plots_dir: str,
    plots_formats: List[str],
    label_rules: List[Tuple[str, str]],
):
    if not main_rows:
        print("[SUMMARY] No main rows. Skipping summary and plots.")
        return

    _apply_paper_rcparams()

    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in main_rows:
        key = (str(r.get("model_name", "")), str(r.get("model_path", "")))
        groups.setdefault(key, []).append(r)

    summary_rows: List[Dict[str, Any]] = []
    raw_labels: List[str] = []

    for (model_name, model_path), gr in groups.items():
        label = label_for_model_name(model_name, model_path, label_rules)

        n_seeds = len({int(float(x.get("seed", 0) or 0)) for x in gr})
        n_total = int(float(gr[0].get("n_images_total", 0) or 0))

        n_used_list = [_safe_float(x.get("n_images_used", "nan")) for x in gr]
        n_miss_list = [_safe_float(x.get("n_images_missing", "nan")) for x in gr]

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
        nu_m, nu_s = _nanmean_std(n_used_list)
        nm_m, nm_s = _nanmean_std(n_miss_list)

        prop_lists: Dict[int, List[float]] = {k: [] for k in range(0, 6)}
        for rr in gr:
            nimg_used = int(float(rr.get("n_images_used", 0) or 0))
            if nimg_used <= 0:
                continue
            for k in range(0, 6):
                c = float(rr.get(f"n_img_correct_{k}", 0) or 0)
                prop_lists[k].append(c / float(nimg_used))

        prop_stats: Dict[int, Tuple[float, float]] = {k: _nanmean_std(prop_lists[k]) for k in range(0, 6)}

        summary_rows.append(
            dict(
                label=label,
                model_name=model_name,
                model_path=model_path,
                n_seeds=int(n_seeds),
                n_images_total=int(n_total),
                n_images_used_mean=float(nu_m),
                n_images_used_std=float(nu_s),
                n_images_missing_mean=float(nm_m),
                n_images_missing_std=float(nm_s),
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

    unique_labels = uniquify_labels(raw_labels)
    for i in range(len(summary_rows)):
        summary_rows[i]["label"] = unique_labels[i]

    summary_rows.sort(key=lambda r: (MODEL_ORDER_INDEX.get(str(r.get("model_name", "")), 10000), str(r.get("model_name", ""))))

    write_csv(summary_csv_path, SUMMARY_CSV_FIELDS, summary_rows)
    print(f"[SUMMARY] Wrote: {summary_csv_path}")

    clean_summary_csv_path = os.path.join(os.path.dirname(summary_csv_path), "ranking_summary_clean_6sf.csv")
    write_clean_summary_csv_6sf(clean_summary_csv_path, summary_rows)

    ensure_dir(plots_dir)

    fmt_set = []
    for f in plots_formats:
        ff = str(f).lower()
        if ff in ("png", "pdf", "svg") and ff not in fmt_set:
            fmt_set.append(ff)
    if not fmt_set:
        fmt_set = ["png"]
    if "png" not in fmt_set:
        fmt_set = ["png"] + fmt_set

    for spec in METRIC_SPECS:
        _save_individual_metric_plot(
            summary_rows=summary_rows,
            spec=spec,
            out_base_no_ext=os.path.join(plots_dir, spec["key"].replace("_mean", "")),
            formats=fmt_set,
        )

    _plot_stacked_distribution(
        summary_rows=summary_rows,
        out_base_no_ext=os.path.join(plots_dir, "correct_count_distribution_out_of_5"),
        formats=fmt_set,
    )

    _plot_all_metric_bars_grid_1row(
        summary_rows=summary_rows,
        out_png_path=os.path.join(plots_dir, "all_metric_bars_grid_1row.png"),
    )

    _plot_paper_combined_metrics_heatmap_variants(summary_rows, plots_dir)

    _plot_paper_metrics_row_2plots_all5(
        summary_rows=summary_rows,
        out_png_path=os.path.join(plots_dir, "paper_metrics_row_2plots_all5.png"),
    )
    _plot_paper_metrics_row_3plots_all5(
        summary_rows=summary_rows,
        out_png_path=os.path.join(plots_dir, "paper_metrics_row_3plots_all5.png"),
    )


# =============================================================================
# DRIVER
# =============================================================================
def run_eval(args):
    ann_images = load_annotation_json(args.annotation_json, args.images_root, max_images=int(args.max_eval_images))
    n_eval = len(ann_images)
    print(f">> Annotated eval images: {n_eval} (max_eval_images={args.max_eval_images}; 0 means all)")
    print(f">> eval_native_size={bool(args.eval_native_size)} | input_size={int(args.input_size)} (used only if eval_native_size=False)")

    invert_models = set([str(x).strip() for x in (args.invert_model or []) if str(x).strip()])
    if invert_models:
        print(f">> Inverting depth for models: {sorted(invert_models)}")

    model_dirs = dict(DEFAULT_MODEL_DIRS)
    overrides = parse_model_dirs_override(args.model_dir)
    if overrides:
        model_dirs.update(overrides)

    ordered = [k for k in DEFAULT_MODEL_ORDER if k in model_dirs]
    extra = [k for k in model_dirs.keys() if k not in ordered]
    model_names = ordered + extra
    if not model_names:
        raise RuntimeError("No models configured. Use --model_dir NAME=PATH.")

    ensure_dir(args.output_root)
    main_csv_path = os.path.join(args.output_root, "ranking_metrics.csv")
    per_image_csv_path = os.path.join(args.output_root, "ranking_per_image.csv")
    summary_csv_path = os.path.join(args.output_root, "ranking_summary.csv")

    label_rules = parse_label_rules(args.label)

    all_main_rows: List[Dict[str, Any]] = []
    all_per_image_rows: List[Dict[str, Any]] = []

    for mn in model_names:
        mp = normalize_path(model_dirs[mn])

        print("\n==================== MODEL ====================")
        print(f">> model_name: {mn}")
        print(f">> model_path: {mp}")
        print("==============================================\n")

        seed_depth_dirs = resolve_depth_dirs_for_model(mp)
        invert_this_model = (mn in invert_models)

        for (seed_int, seed_label, depth_dir) in seed_depth_dirs:
            print(f">> Evaluating: {mn} | {seed_label} | depth_dir={depth_dir}")

            main_row, per_rows = eval_one_model_one_seed_from_depth_dir(
                model_name=mn,
                model_path=mp,
                seed_int=seed_int,
                seed_label=seed_label,
                depth_dir=depth_dir,
                ann_images=ann_images,
                eval_native_size=bool(args.eval_native_size),
                input_size=int(args.input_size),
                rank1_is_closest=bool(args.rank1_is_closest),
                pred_higher_is_farther=bool(args.pred_higher_is_farther),
                tie_epsilon=float(args.tie_epsilon),
                strict_missing=bool(args.strict_missing),
                invert_depth=bool(invert_this_model),
            )

            for rr in per_rows:
                rr["images_root"] = str(normalize_path(args.images_root))
                rr["annotation_json"] = str(normalize_path(args.annotation_json))

            main_row["images_root"] = str(normalize_path(args.images_root))
            main_row["annotation_json"] = str(normalize_path(args.annotation_json))

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
                f">> {mn} | {seed_label} | used={main_row['n_images_used']}/{main_row['n_images_total']} "
                f"(missing={main_row['n_images_missing']}) | "
                f"mean_point_rank_acc={main_row['mean_point_rank_accuracy']:.4f} | "
                f"pairwise_acc={main_row['pairwise_accuracy_mean']:.4f} | "
                f"exact_match={main_row['exact_match_rate']:.4f} | "
                f"kendall_tau={main_row['kendall_tau_mean']:.4f} | "
                f"spearman_rho={main_row['spearman_rho_mean']:.4f} | {dist}"
            )

            all_main_rows.append(main_row)
            all_per_image_rows.extend(per_rows)

            if bool(args.flush_csv_each_seed):
                write_csv(main_csv_path, MAIN_CSV_FIELDS, all_main_rows)
                write_csv(per_image_csv_path, PER_IMAGE_CSV_FIELDS, all_per_image_rows)

    write_csv(main_csv_path, MAIN_CSV_FIELDS, all_main_rows)
    write_csv(per_image_csv_path, PER_IMAGE_CSV_FIELDS, all_per_image_rows)

    print(f"\n[CSV] Wrote main:     {main_csv_path}")
    print(f"[CSV] Wrote per-img: {per_image_csv_path}")

    if bool(args.make_plots) or bool(args.make_summary):
        plots_formats = [x.strip().lower() for x in str(args.plots_format).split(",") if x.strip()]
        plots_formats = [x for x in plots_formats if x in ("png", "pdf", "svg")]
        if not plots_formats:
            plots_formats = ["png"]

        generate_summary_and_plots(
            main_rows=all_main_rows,
            summary_csv_path=summary_csv_path,
            plots_dir=os.path.join(args.output_root, "plots"),
            plots_formats=plots_formats,
            label_rules=label_rules,
        )


# =============================================================================
# ARGS
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute depth-ranking metrics from already-computed depth PNG folders (native-size aware)."
    )

    parser.add_argument("--images_root", type=str, default=DEFAULT_IMAGES_ROOT)
    parser.add_argument("--annotation_json", type=str, default=DEFAULT_ANNOTATION_JSON)

    parser.add_argument(
        "--eval_native_size",
        type=str2bool,
        default=True,
        help="True: evaluate at each depth PNG native shape. False: resize all depth maps to --input_size first.",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=DEFAULT_INPUT_SIZE,
        help="Used only if --eval_native_size=False",
    )

    parser.add_argument("--max_eval_images", type=int, default=0, help="0 means all annotated images")

    parser.add_argument("--rank1_is_closest", type=str2bool, default=True)
    parser.add_argument("--pred_higher_is_farther", type=str2bool, default=True)
    parser.add_argument("--tie_epsilon", type=float, default=0.0)

    parser.add_argument(
        "--model_dir",
        type=str,
        action="append",
        default=[],
        help='Add or override model dir (repeatable): --model_dir "PPSNet=/path/to/depths"',
    )

    parser.add_argument(
        "--invert_model",
        type=str,
        action="append",
        default=[],
        help='Repeatable model names to invert depth as (1 - depth), for example --invert_model PPSNet',
    )

    parser.add_argument("--output_root", type=str, default="./ranking_eval_from_preds")
    parser.add_argument("--flush_csv_each_seed", type=str2bool, default=True)

    parser.add_argument("--strict_missing", type=str2bool, default=False, help="If True, stop on missing depth PNG")

    parser.add_argument("--make_plots", type=str2bool, default=True)
    parser.add_argument("--make_summary", type=str2bool, default=True)
    parser.add_argument("--plots_format", type=str, default="png", help="Comma-separated: png,pdf,svg")

    parser.add_argument(
        "--label",
        type=str,
        nargs="*",
        default=[],
        help='Custom plot label rules: --label "match_substring=Nice Label"',
    )

    args = parser.parse_args()

    args.images_root = normalize_path(args.images_root)
    args.annotation_json = normalize_path(args.annotation_json)
    args.output_root = normalize_path(args.output_root)

    if not os.path.isdir(args.images_root):
        raise FileNotFoundError(f"--images_root not found: {args.images_root}")
    if not os.path.isfile(args.annotation_json):
        raise FileNotFoundError(f"--annotation_json not found: {args.annotation_json}")

    if int(args.input_size) <= 0:
        raise ValueError("--input_size must be > 0")
    if int(args.max_eval_images) < 0:
        raise ValueError("--max_eval_images must be >= 0")

    return args


def main():
    args = parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()