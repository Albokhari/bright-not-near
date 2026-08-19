#!/usr/bin/env python3
"""
merge_pass_consensus_to_gt.py

Merge PASS images from multiple consensus outputs into:
  1) a single image folder (copied)
  2) a single JSON in GT-style format (same schema as depth_annotation_gt)

Inputs per dataset:
  - GT annotation JSON (for x/y coords and point_id)
  - consensus summary JSON from rank_consensus_paper.py (must include decisions)
  - images dir used in rank_consensus_paper.py (script looks in <images_dir>/pass first)

Why summary JSON?
  We need the consensus_order (winning order) for each PASS image, which is stored in the
  summary JSON produced by rank_consensus_paper.py when --summary-json is used.

Example:
python merge_pass_consensus_to_gt.py \
  --dataset Annotator3  "./Annotator3_data/ranking/gt_annotator3.json"   "./Annotator3_data/annotator3_consensus_summary.json"   "./Annotator3_data/ranking/200" \
  --dataset Annotator2   "./Annotator2_data/ranking/gt_annotator2.json"     "./Annotator2_data/annotator2_consensus_summary.json"     "./Annotator2_data/ranking/100" \
  --dataset Annotator4   "./Tim_data/ranking/gt_annotator4.json"     "./Tim_data/annotator4_consensus_summary.json"     "./Tim_data/ranking/val" \
  --dataset Annotator1 "./annotator1_data/ranking/gt_annotator1.json" "./annotator1_data/annotator1_consensus_summary.json" "./annotator1_data/ranking/test" \
  --out-images-dir "./merged_pass/images" \
  --out-json "./merged_pass/merged_pass_consensus_gt.json"

Notes:
- Images are COPIED (never moved).
- If a PASS image is not found in <images_dir>/pass, it also searches <images_dir>/ and <images_dir>/fail.
- Duplicate filenames across datasets are rare, but handled via --duplicate-mode.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Helpers
# ============================================================
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to read JSON: {path} ({e})") from e


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def validate_order_list(v: Any, *, dataset: str, image_name: str) -> List[int]:
    if not isinstance(v, list) or not all(isinstance(x, int) for x in v):
        raise ValueError(f"[{dataset}] {image_name}: consensus_order must be a list[int], got {v!r}")
    if len(v) == 0:
        raise ValueError(f"[{dataset}] {image_name}: consensus_order is empty")
    if len(set(v)) != len(v):
        raise ValueError(f"[{dataset}] {image_name}: consensus_order contains duplicates: {v}")
    return v


def find_image_file(images_dir: Path, filename: str) -> Optional[Path]:
    """
    Search order:
      1) <images_dir>/pass/<filename>
      2) <images_dir>/<filename>
      3) <images_dir>/fail/<filename>
    """
    candidates = [
        images_dir / "pass" / filename,
        images_dir / filename,
        images_dir / "fail" / filename,
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def build_gt_lookup(gt_json: dict, dataset_name: str) -> Tuple[Dict[str, dict], int, Optional[str]]:
    """
    Returns:
      filename -> full image record
      numPoints
      project
    """
    if not isinstance(gt_json, dict):
        raise ValueError(f"[{dataset_name}] GT JSON root must be an object")

    images = gt_json.get("images")
    if not isinstance(images, list):
        raise ValueError(f"[{dataset_name}] GT JSON missing valid 'images' list")

    num_points = gt_json.get("numPoints")
    if not isinstance(num_points, int) or num_points <= 0:
        raise ValueError(f"[{dataset_name}] GT JSON invalid numPoints: {num_points!r}")

    project = gt_json.get("project")
    project_str = str(project) if project is not None else None

    lookup: Dict[str, dict] = {}
    for img in images:
        if not isinstance(img, dict):
            raise ValueError(f"[{dataset_name}] Invalid image record in GT JSON")
        fn = img.get("filename")
        if not isinstance(fn, str) or not fn.strip():
            raise ValueError(f"[{dataset_name}] GT image record missing valid filename")
        fn = fn.strip()
        if fn in lookup:
            raise ValueError(f"[{dataset_name}] Duplicate filename in GT JSON: {fn}")
        lookup[fn] = img

    return lookup, num_points, project_str


def build_pass_consensus_lookup(summary_json: dict, dataset_name: str) -> Dict[str, List[int]]:
    """
    Reads the summary JSON from rank_consensus_paper.py and returns:
      filename -> consensus_order (list[int]) for PASS images only
    """
    if not isinstance(summary_json, dict):
        raise ValueError(f"[{dataset_name}] Summary JSON root must be an object")

    decisions = summary_json.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError(f"[{dataset_name}] Summary JSON missing valid 'decisions' list")

    out: Dict[str, List[int]] = {}

    for d in decisions:
        if not isinstance(d, dict):
            raise ValueError(f"[{dataset_name}] Invalid decision record in summary JSON")

        image_name = d.get("image_name")
        decision = d.get("decision")
        consensus_order = d.get("consensus_order")

        if not isinstance(image_name, str) or not image_name.strip():
            raise ValueError(f"[{dataset_name}] Decision missing valid image_name")

        if str(decision).upper() != "PASS":
            continue

        order = validate_order_list(consensus_order, dataset=dataset_name, image_name=image_name)
        if image_name in out:
            raise ValueError(f"[{dataset_name}] Duplicate PASS decision for image: {image_name}")

        out[image_name] = order

    return out


def rewrite_points_with_consensus_rank(
    gt_image_record: dict,
    consensus_order: List[int],
    dataset_name: str,
    image_name: str,
) -> dict:
    """
    Use GT coords/point IDs, but replace rank according to consensus_order.
    Output remains GT-style image record with same width/height/filename/points structure.
    """
    points = gt_image_record.get("points")
    if not isinstance(points, list) or len(points) == 0:
        raise ValueError(f"[{dataset_name}] {image_name}: GT points missing/empty")

    point_lookup: Dict[int, dict] = {}
    for p in points:
        if not isinstance(p, dict):
            raise ValueError(f"[{dataset_name}] {image_name}: GT point is not an object")
        pid = p.get("point_id")
        if not isinstance(pid, int):
            raise ValueError(f"[{dataset_name}] {image_name}: GT point_id must be int")
        if pid in point_lookup:
            raise ValueError(f"[{dataset_name}] {image_name}: duplicate GT point_id={pid}")
        point_lookup[pid] = p

    gt_ids = set(point_lookup.keys())
    order_ids = set(consensus_order)
    if gt_ids != order_ids:
        missing = sorted(gt_ids - order_ids)
        extra = sorted(order_ids - gt_ids)
        raise ValueError(
            f"[{dataset_name}] {image_name}: consensus_order point_ids mismatch GT.\n"
            f"  missing_from_order={missing}\n"
            f"  extra_in_order={extra}\n"
            f"  gt_point_ids={sorted(gt_ids)}\n"
            f"  consensus_order={consensus_order}"
        )

    rank_by_pid = {pid: rank for rank, pid in enumerate(consensus_order, start=1)}

    # Preserve original point list order; only overwrite rank values
    new_points: List[dict] = []
    for p in points:
        new_p = dict(p)
        new_p["rank"] = rank_by_pid[p["point_id"]]
        new_points.append(new_p)

    new_img = dict(gt_image_record)
    new_img["points"] = new_points
    return new_img


def resolve_output_filename(
    filename: str,
    dataset_name: str,
    used_names: set,
    duplicate_mode: str,
) -> str:
    """
    duplicate_mode:
      - error : raise if duplicate filename across datasets
      - prefix: prefix dataset name (only when duplicate occurs)
    """
    if filename not in used_names:
        return filename

    if duplicate_mode == "error":
        raise ValueError(
            f"Duplicate filename across datasets: {filename}. "
            f"Use --duplicate-mode prefix to auto-prefix dataset names."
        )

    # prefix mode
    candidate = f"{dataset_name}__{filename}"
    if candidate in used_names:
        raise ValueError(
            f"Duplicate even after prefixing: {candidate}. "
            f"Please rename manually or use disjoint filenames."
        )
    return candidate


# ============================================================
# Main merge logic
# ============================================================
def merge_datasets(
    datasets: List[Tuple[str, Path, Path, Path]],
    out_images_dir: Path,
    out_json: Path,
    duplicate_mode: str,
    project_name: str,
    annotator_name: str,
    json_type: str,
    strict_missing_files: bool,
    strict_missing_gt: bool,
) -> None:
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    merged_images: List[dict] = []
    used_output_filenames = set()

    num_points_values = []
    projects_seen = []
    per_dataset_counts = Counter()
    per_dataset_missing_files = Counter()
    per_dataset_missing_gt = Counter()

    for dataset_name, gt_path, summary_path, images_dir in datasets:
        gt_json = load_json(gt_path)
        summary_json = load_json(summary_path)

        gt_lookup, num_points, project = build_gt_lookup(gt_json, dataset_name=dataset_name)
        pass_lookup = build_pass_consensus_lookup(summary_json, dataset_name=dataset_name)

        num_points_values.append(num_points)
        if project:
            projects_seen.append(project)

        for image_name, consensus_order in sorted(pass_lookup.items()):
            gt_img = gt_lookup.get(image_name)
            if gt_img is None:
                per_dataset_missing_gt[dataset_name] += 1
                msg = f"[{dataset_name}] PASS image not found in GT JSON: {image_name}"
                if strict_missing_gt:
                    raise ValueError(msg)
                else:
                    print(f"⚠️ {msg}", file=sys.stderr)
                    continue

            src_img = find_image_file(images_dir, image_name)
            if src_img is None:
                per_dataset_missing_files[dataset_name] += 1
                msg = f"[{dataset_name}] PASS image file not found in images dir: {image_name} (searched pass/, root, fail/)"
                if strict_missing_files:
                    raise FileNotFoundError(msg)
                else:
                    print(f"⚠️ {msg}", file=sys.stderr)
                    continue

            out_filename = resolve_output_filename(
                filename=image_name,
                dataset_name=dataset_name,
                used_names=used_output_filenames,
                duplicate_mode=duplicate_mode,
            )
            used_output_filenames.add(out_filename)

            # Copy image
            dst_img = out_images_dir / out_filename
            if dst_img.exists():
                dst_img.unlink()
            shutil.copy2(src_img, dst_img)

            # Build GT-style image record with consensus ranks
            new_img = rewrite_points_with_consensus_rank(
                gt_image_record=gt_img,
                consensus_order=consensus_order,
                dataset_name=dataset_name,
                image_name=image_name,
            )

            # Update filename if prefixed due to duplicate handling
            new_img["filename"] = out_filename

            merged_images.append(new_img)
            per_dataset_counts[dataset_name] += 1

    # numPoints consistency
    if len(set(num_points_values)) > 1:
        raise ValueError(f"numPoints mismatch across GT JSONs: {num_points_values}")
    merged_num_points = num_points_values[0] if num_points_values else 0

    # Keep deterministic output ordering
    merged_images.sort(key=lambda x: x.get("filename", ""))

    payload = {
        "type": json_type,
        "annotator": annotator_name,
        "project": project_name,
        "numPoints": merged_num_points,
        "createdAt": now_utc_iso(),
        "images": merged_images,
    }

    save_json(out_json, payload)

    # Console summary
    total_pass = sum(per_dataset_counts.values())
    print("\n✅ Merge complete")
    print(f"📁 Merged PASS images folder : {out_images_dir}")
    print(f"🧾 Merged GT-style JSON      : {out_json}")
    print(f"🧮 Total merged PASS images  : {total_pass}")
    print()
    print("Per-dataset counts:")
    for dataset_name, _, _, _ in datasets:
        print(
            f"  - {dataset_name}: "
            f"merged={per_dataset_counts[dataset_name]}, "
            f"missing_file={per_dataset_missing_files[dataset_name]}, "
            f"missing_gt={per_dataset_missing_gt[dataset_name]}"
        )


# ============================================================
# CLI
# ============================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Merge PASS images from multiple consensus outputs into one folder + one GT-style JSON."
    )

    p.add_argument(
        "--dataset",
        action="append",
        nargs=4,
        metavar=("NAME", "GT_JSON", "SUMMARY_JSON", "IMAGES_DIR"),
        required=True,
        help=(
            "Add one dataset as: NAME GT_JSON SUMMARY_JSON IMAGES_DIR. "
            "Repeat this flag 4 times (or more)."
        ),
    )

    p.add_argument(
        "--out-images-dir",
        required=True,
        help="Output folder where all PASS images will be copied.",
    )
    p.add_argument(
        "--out-json",
        required=True,
        help="Output merged GT-style JSON path.",
    )

    p.add_argument(
        "--project-name",
        default="merged_consensus_pass",
        help="Top-level 'project' field for the merged JSON.",
    )
    p.add_argument(
        "--annotator-name",
        default="consensus_merged",
        help="Top-level 'annotator' field for the merged JSON.",
    )
    p.add_argument(
        "--json-type",
        default="depth_annotation_gt",
        help="Top-level 'type' field (default keeps GT-style compatibility).",
    )

    p.add_argument(
        "--duplicate-mode",
        choices=["error", "prefix"],
        default="error",
        help="How to handle duplicate filenames across datasets.",
    )

    p.add_argument(
        "--allow-missing-files",
        action="store_true",
        help="Skip PASS images whose files are missing instead of failing.",
    )
    p.add_argument(
        "--allow-missing-gt",
        action="store_true",
        help="Skip PASS images missing in GT JSON instead of failing.",
    )

    return p


def main() -> int:
    args = build_parser().parse_args()

    datasets: List[Tuple[str, Path, Path, Path]] = []
    for row in args.dataset:
        name, gt_json, summary_json, images_dir = row
        gt_path = Path(gt_json).expanduser().resolve()
        summary_path = Path(summary_json).expanduser().resolve()
        img_dir = Path(images_dir).expanduser().resolve()

        if not gt_path.exists() or not gt_path.is_file():
            print(f"❌ GT JSON not found: {gt_path}", file=sys.stderr)
            return 1
        if not summary_path.exists() or not summary_path.is_file():
            print(f"❌ Summary JSON not found: {summary_path}", file=sys.stderr)
            return 1
        if not img_dir.exists() or not img_dir.is_dir():
            print(f"❌ Images dir not found: {img_dir}", file=sys.stderr)
            return 1

        datasets.append((name, gt_path, summary_path, img_dir))

    out_images_dir = Path(args.out_images_dir).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve()

    try:
        merge_datasets(
            datasets=datasets,
            out_images_dir=out_images_dir,
            out_json=out_json,
            duplicate_mode=args.duplicate_mode,
            project_name=args.project_name,
            annotator_name=args.annotator_name,
            json_type=args.json_type,
            strict_missing_files=not args.allow_missing_files,
            strict_missing_gt=not args.allow_missing_gt,
        )
        return 0
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())