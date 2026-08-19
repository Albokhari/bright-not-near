#!/usr/bin/env python3
"""
Ring-based Point Sampler for Depth Annotation (Ranker JSON) — Multi-threaded batch

Key behavior:
- Samples N points from concentric rings around the image center.
- Avoids invalid pixels:
  - black / masked background (<= black_threshold)
  - transparent pixels (alpha == 0, if present)
  - bright light reflections (grayscale > reflect_threshold)
- Batch mode is multi-threaded.
- JSON is the primary output; visualization saving is optional and OFF by default.

JSON format:
{
  "type": "depth_annotation_ranker",
  "project": "project_1",
  "numPoints": 5,
  "createdAt": "2026-02-03T02:40:14.373Z",
  "images": [
    {
      "filename": "...",
      "width": 616,
      "height": 530,
      "points": [{"point_id":1,"x":...,"y":...,"rank":null}, ...]
    },
    ...
  ]
}
"""

import os
import json
import random
from datetime import datetime, timezone
from typing import Tuple, List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
from PIL import Image


# -----------------------------------------------------------------------------
# Masks / Geometry
# -----------------------------------------------------------------------------

def _to_gray_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convert image array to grayscale (0..255 range) as float32.
    Supports:
      - (H,W) grayscale
      - (H,W,3) RGB
      - (H,W,4) RGBA
      - (H,W,2) LA
    """
    if image.ndim == 2:
        return image.astype(np.float32)

    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    c = image.shape[2]
    if c == 1:
        return image[..., 0].astype(np.float32)
    if c == 2:
        # LA
        return image[..., 0].astype(np.float32)
    if c >= 3:
        rgb = image[..., :3].astype(np.float32)
        # ITU-R BT.601 luma
        return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    raise ValueError(f"Unsupported channel count: {c}")


def get_valid_mask(
    image: np.ndarray,
    black_threshold: int = 10,
    reflect_threshold: Optional[int] = 220
) -> np.ndarray:
    """
    Create a boolean mask of valid pixels.

    Valid pixel criteria:
      - Not black:
          * grayscale > black_threshold for grayscale/LA
          * OR any RGB channel > black_threshold for RGB/RGBA
      - Not a bright reflection: grayscale <= reflect_threshold (if enabled)
      - If alpha exists, alpha > 0
    """
    if image.ndim == 2:
        gray = image.astype(np.float32)
        non_black = gray > black_threshold
        alpha_ok = np.ones_like(non_black, dtype=bool)
    elif image.ndim == 3:
        h, w, c = image.shape
        if c == 1:
            gray = image[..., 0].astype(np.float32)
            non_black = gray > black_threshold
            alpha_ok = np.ones((h, w), dtype=bool)
        elif c == 2:
            # LA
            gray = image[..., 0].astype(np.float32)
            non_black = gray > black_threshold
            alpha = image[..., 1].astype(np.float32)
            alpha_ok = alpha > 0
        elif c == 3:
            rgb = image[..., :3]
            non_black = np.any(rgb > black_threshold, axis=2)
            alpha_ok = np.ones((h, w), dtype=bool)
        else:
            # RGBA (or more; use first 4)
            rgb = image[..., :3]
            non_black = np.any(rgb > black_threshold, axis=2)
            alpha = image[..., 3].astype(np.float32)
            alpha_ok = alpha > 0
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    if reflect_threshold is None:
        not_reflection = np.ones_like(non_black, dtype=bool)
    else:
        gray = _to_gray_uint8(image)
        not_reflection = gray <= float(reflect_threshold)

    return non_black & not_reflection & alpha_ok


def compute_distance_from_center(height: int, width: int) -> np.ndarray:
    """Euclidean distance of each pixel from image center."""
    center_y, center_x = height / 2.0, width / 2.0
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    return np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)


def create_ring_bounds_from_valid(
    distances: np.ndarray,
    valid_mask: np.ndarray,
    num_rings: int
) -> List[Tuple[float, float]]:
    """Create [inner, outer) bounds for num_rings using max distance within valid region."""
    valid_distances = distances[valid_mask]
    if valid_distances.size == 0:
        raise ValueError("No valid pixels found (after black/reflection/alpha filtering).")

    max_distance = float(np.max(valid_distances))
    ring_thickness = max_distance / float(num_rings)

    bounds: List[Tuple[float, float]] = []
    for i in range(num_rings):
        inner = i * ring_thickness
        outer = (i + 1) * ring_thickness
        bounds.append((inner, outer))
    return bounds


# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------

def _pick_random_coord_from_mask(mask: np.ndarray, rng: random.Random) -> Optional[Tuple[int, int]]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    i = rng.randrange(ys.size)
    return int(xs[i]), int(ys[i])


def _sample_point_for_ring(
    distances: np.ndarray,
    valid_mask: np.ndarray,
    inner: float,
    outer: float,
    used_linear: set,
    rng: random.Random,
    fallback_topk: int = 2000
) -> Optional[Tuple[int, int]]:
    """
    Try to sample within the ring. If empty, fall back to valid pixels closest to ring midpoint.
    Ensures no duplicates via used_linear.
    """
    h, w = valid_mask.shape
    ring_mask = valid_mask & (distances >= inner) & (distances < outer)

    if used_linear:
        ys, xs = np.where(ring_mask)
        if ys.size > 0:
            lin = ys.astype(np.int64) * w + xs.astype(np.int64)
            keep = np.array([int(v) not in used_linear for v in lin], dtype=bool)
            if np.any(keep):
                ys2 = ys[keep]
                xs2 = xs[keep]
                j = rng.randrange(ys2.size)
                return int(xs2[j]), int(ys2[j])
    else:
        pt = _pick_random_coord_from_mask(ring_mask, rng)
        if pt is not None:
            return pt

    # Fallback: choose among valid pixels closest to ring midpoint distance
    all_ys, all_xs = np.where(valid_mask)
    if all_ys.size == 0:
        return None

    lin_all = all_ys.astype(np.int64) * w + all_xs.astype(np.int64)
    if used_linear:
        keep = np.array([int(v) not in used_linear for v in lin_all], dtype=bool)
        if not np.any(keep):
            return None
        all_ys = all_ys[keep]
        all_xs = all_xs[keep]

    mid = 0.5 * (inner + outer)
    d = distances[all_ys, all_xs]
    diffs = np.abs(d - mid)

    k = min(int(fallback_topk), diffs.size)
    if k <= 0:
        return None

    idxs = np.argpartition(diffs, k - 1)[:k]
    chosen = rng.choice(idxs.tolist())
    return int(all_xs[chosen]), int(all_ys[chosen])


def sample_points_from_rings(
    image: np.ndarray,
    num_rings: int = 5,
    black_threshold: int = 10,
    reflect_threshold: Optional[int] = 220,
    seed: Optional[int] = None
) -> List[Tuple[int, int]]:
    """
    Sample one point per ring, with robust fallback if a ring has no valid pixels.
    Returns up to num_rings points (exactly num_rings if enough valid pixels exist).
    """
    rng = random.Random(seed)

    h, w = image.shape[:2]
    valid_mask = get_valid_mask(image, black_threshold=black_threshold, reflect_threshold=reflect_threshold)

    distances = compute_distance_from_center(h, w)
    bounds = create_ring_bounds_from_valid(distances, valid_mask, num_rings=num_rings)

    used_linear: set = set()
    points: List[Tuple[int, int]] = []

    for (inner, outer) in bounds:
        pt = _sample_point_for_ring(distances, valid_mask, inner, outer, used_linear, rng)
        if pt is None:
            break
        x, y = pt
        used_linear.add(int(y) * w + int(x))
        points.append((x, y))

    # Fill remaining (if needed) from anywhere valid, still no duplicates
    if len(points) < num_rings:
        all_ys, all_xs = np.where(valid_mask)
        if all_ys.size == 0:
            return points

        lin_all = all_ys.astype(np.int64) * w + all_xs.astype(np.int64)
        keep = np.array([int(v) not in used_linear for v in lin_all], dtype=bool)
        all_ys = all_ys[keep]
        all_xs = all_xs[keep]

        idxs = list(range(all_ys.size))
        rng.shuffle(idxs)
        for idx in idxs:
            if len(points) >= num_rings:
                break
            x = int(all_xs[idx])
            y = int(all_ys[idx])
            used_linear.add(int(y) * w + int(x))
            points.append((x, y))

    return points


# -----------------------------------------------------------------------------
# Visualization (optional)
# -----------------------------------------------------------------------------

def visualize_rings_and_points(
    image: np.ndarray,
    points: List[Tuple[int, int]],
    point_labels: Optional[List[int]] = None,
    output_path: Optional[str] = None,
    show: bool = False,
    show_rings: bool = False,
    num_rings: Optional[int] = None,
    black_threshold: int = 10,
    reflect_threshold: Optional[int] = 220,
):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    valid_mask = get_valid_mask(image, black_threshold=black_threshold, reflect_threshold=reflect_threshold)
    distances = compute_distance_from_center(h, w)
    valid_distances = distances[valid_mask]
    max_distance = float(np.max(valid_distances)) if valid_distances.size > 0 else float(np.sqrt(h * h + w * w) / 2.0)

    ring_count = int(num_rings) if num_rings is not None else max(1, len(points))
    ring_thickness = max_distance / float(ring_count)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(image)

    if show_rings:
        for i in range(ring_count + 1):
            radius = i * ring_thickness
            ax.add_patch(
                Circle((cx, cy), radius, fill=False, edgecolor="white",
                       linewidth=1.5, linestyle="--", alpha=0.7)
            )

    point_color = "#E8A598"
    for i, (x, y) in enumerate(points):
        label = point_labels[i] if point_labels is not None else (i + 1)
        ax.scatter(x, y, c=point_color, s=300, edgecolors="white", linewidth=3, zorder=5)
        ax.text(x, y, f"{label}", fontsize=14, fontweight="bold", color="white",
                ha="center", va="center", zorder=6)

    ax.set_title(f"{len(points)} points sampled", fontsize=14)
    ax.axis("off")
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")

    if show:
        plt.show()
    else:
        plt.close(fig)


# -----------------------------------------------------------------------------
# Batch processing + JSON
# -----------------------------------------------------------------------------

def _iso_utc_now_ms() -> str:
    # Example: 2026-02-03T02:40:14.373Z
    now = datetime.now(timezone.utc)
    return now.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _list_images(input_folder: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    files = [f for f in os.listdir(input_folder) if os.path.splitext(f)[1].lower() in exts]
    files.sort()
    return files


def _process_one_image(
    idx: int,
    filename: str,
    input_folder: str,
    num_rings: int,
    black_threshold: int,
    reflect_threshold: Optional[int],
    base_seed: Optional[int],
    save_viz: bool,
    viz_folder: Optional[str],
    show_rings: bool,
) -> Tuple[int, Dict[str, Any]]:
    """
    Returns (idx, image_entry_dict). idx is used to preserve stable JSON ordering.
    """
    image_path = os.path.join(input_folder, filename)

    img_seed = (base_seed + idx) if base_seed is not None else None
    rng = random.Random(img_seed)

    # Load
    image = np.array(Image.open(image_path))
    h, w = image.shape[:2]

    # Sample
    points_xy = sample_points_from_rings(
        image=image,
        num_rings=num_rings,
        black_threshold=black_threshold,
        reflect_threshold=reflect_threshold,
        seed=img_seed,
    )

    # Shuffle point IDs so numbering isn't tied to ring order
    labels = list(range(1, len(points_xy) + 1))
    rng.shuffle(labels)

    point_dicts = [{"point_id": labels[i], "x": int(x), "y": int(y), "rank": None}
                   for i, (x, y) in enumerate(points_xy)]
    point_dicts.sort(key=lambda d: d["point_id"])

    # Optional viz
    if save_viz and viz_folder is not None:
        try:
            out_vis_path = os.path.join(viz_folder, filename)
            # For viz alignment, create labels in the same order as points_xy
            visualize_rings_and_points(
                image=image,
                points=points_xy,
                point_labels=labels,
                output_path=out_vis_path,
                show=False,
                show_rings=show_rings,
                num_rings=num_rings,
                black_threshold=black_threshold,
                reflect_threshold=reflect_threshold,
            )
        except Exception as e:
            # Never fail the batch for viz issues (JSON is primary)
            print(f"[viz warning] {filename}: {e}")

    entry = {
        "filename": filename,
        "width": int(w),
        "height": int(h),
        "points": point_dicts
    }
    return idx, entry


def process_all_images_threaded(
    input_folder: str,
    num_rings: int = 5,
    black_threshold: int = 10,
    reflect_threshold: Optional[int] = 220,
    seed: Optional[int] = None,
    workers: Optional[int] = None,
    save_viz: bool = False,
    viz_folder: Optional[str] = None,
    show_rings: bool = False,
) -> List[Dict[str, Any]]:
    """
    Multi-threaded folder processing. Preserves deterministic ordering in JSON
    according to sorted filenames.
    """
    files = _list_images(input_folder)
    n = len(files)
    if n == 0:
        return []

    if save_viz:
        if viz_folder is None:
            viz_folder = os.path.join(input_folder, "sampled_viz")
        os.makedirs(viz_folder, exist_ok=True)

    results: List[Optional[Dict[str, Any]]] = [None] * n
    lock = threading.Lock()
    done = 0

    max_workers = workers if workers is not None else (os.cpu_count() or 8)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = []
        for idx, fn in enumerate(files):
            futs.append(ex.submit(
                _process_one_image,
                idx, fn, input_folder,
                num_rings, black_threshold, reflect_threshold, seed,
                save_viz, viz_folder, show_rings
            ))

        for fut in as_completed(futs):
            idx, entry = fut.result()
            results[idx] = entry
            with lock:
                done += 1
                print(f"[{done}/{n}] Processed: {entry['filename']}")

    # mypy-friendly: filter out Nones (shouldn't happen unless exceptions were swallowed)
    return [r for r in results if r is not None]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sample points from concentric rings (Ranker JSON, threaded batch)")
    parser.add_argument("input", help="Path to input image OR folder of images")

    # Sampling params
    parser.add_argument("--rings", type=int, default=5, help="Number of points/rings (default: 5)")
    parser.add_argument("--threshold", type=int, default=10, help="Black pixel threshold (default: 10)")
    parser.add_argument("--reflect-threshold", type=int, default=220,
                        help="Reflection threshold on grayscale (default: 220). Pixels > this are invalid.")
    parser.add_argument("--disable-reflect-filter", action="store_true", help="Disable reflection filtering entirely")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--project", type=str, default="project_1", help="Project name for JSON")

    # JSON output
    parser.add_argument("--save-json", type=str, default=None,
                        help="Path to save JSON output (default: points.json in input folder for batch, or alongside image for single)")

    # Threading
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker threads for batch (default: CPU count)")

    # Optional visualization saving (OFF by default)
    parser.add_argument("--save-viz", action="store_true",
                        help="Save visualization images (OFF by default; JSON is always produced).")
    parser.add_argument("--viz-dir", type=str, default=None,
                        help="Folder to save visualizations (default: <input_folder>/sampled_viz when --save-viz)")
    parser.add_argument("--show-rings", action="store_true", help="Draw ring boundaries in visualization")

    # Single image viewing
    parser.add_argument("--no-show", action="store_true", help="Don't display the visualization (single image)")

    args = parser.parse_args()

    reflect_thr: Optional[int] = None if args.disable_reflect_filter else int(args.reflect_threshold)

    if os.path.isdir(args.input):
        # Batch (threaded)
        images_list = process_all_images_threaded(
            input_folder=args.input,
            num_rings=args.rings,
            black_threshold=args.threshold,
            reflect_threshold=reflect_thr,
            seed=args.seed,
            workers=args.workers,
            save_viz=args.save_viz,
            viz_folder=args.viz_dir,
            show_rings=args.show_rings,
        )

        json_path = args.save_json or os.path.join(args.input, "points.json")

        payload = {
            "type": "depth_annotation_ranker",
            "project": args.project,
            "numPoints": int(args.rings),
            "createdAt": _iso_utc_now_ms(),
            "images": images_list
        }

        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"\nProcessed {len(images_list)} images")
        print(f"Saved JSON to: {json_path}")
        if args.save_viz:
            viz_dir = args.viz_dir or os.path.join(args.input, "sampled_viz")
            print(f"Saved visualizations to: {viz_dir}")

    else:
        # Single image
        image = np.array(Image.open(args.input))
        h, w = image.shape[:2]

        points = sample_points_from_rings(
            image=image,
            num_rings=args.rings,
            black_threshold=args.threshold,
            reflect_threshold=reflect_thr,
            seed=args.seed,
        )

        rng = random.Random(args.seed)
        labels = list(range(1, len(points) + 1))
        rng.shuffle(labels)

        print(f"Image: {args.input}")
        print(f"Image size: {w} x {h}")
        print(f"Sampled {len(points)} points (target {args.rings}):")
        for i, (x, y) in enumerate(points):
            print(f"  Point {labels[i]}: ({x}, {y})")

        # JSON for single image
        if args.save_json:
            point_dicts = [{"point_id": labels[i], "x": int(x), "y": int(y), "rank": None}
                           for i, (x, y) in enumerate(points)]
            point_dicts.sort(key=lambda d: d["point_id"])

            payload = {
                "type": "depth_annotation_ranker",
                "project": args.project,
                "numPoints": int(args.rings),
                "createdAt": _iso_utc_now_ms(),
                "images": [{
                    "filename": os.path.basename(args.input),
                    "width": int(w),
                    "height": int(h),
                    "points": point_dicts
                }]
            }

            with open(args.save_json, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"Saved JSON to: {args.save_json}")

        # Optional viz: show (single) and/or save (if --save-viz)
        out_vis_path = None
        if args.save_viz:
            if args.viz_dir is not None:
                os.makedirs(args.viz_dir, exist_ok=True)
                out_vis_path = os.path.join(args.viz_dir, os.path.basename(args.input))
            else:
                # default next to image
                out_vis_path = os.path.splitext(args.input)[0] + "_viz.png"

        visualize_rings_and_points(
            image=image,
            points=points,
            point_labels=labels,
            output_path=out_vis_path,
            show=(not args.no_show),
            show_rings=args.show_rings,
            num_rings=args.rings,
            black_threshold=args.threshold,
            reflect_threshold=reflect_thr,
        )
