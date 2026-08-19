#!/usr/bin/env python3
"""
rank_consensus_paper.py

Aggregate multiple depth-ranking annotation JSON files (from your ranking app),
compute consensus per image, split images into PASS / FAIL, and export a paper-ready CSV.

PASS rule (exact-order consensus):
- For each image, each annotator contributes a ranking order:
    order = tuple(point_id sorted by rank ascending)
- PASS if the same exact order appears at least `threshold` times
  (e.g., threshold=3 for 3/4 majority).

Outputs:
1) Moves (or copies) images to:
      <images_dir>/pass/
      <images_dir>/fail/
2) CSV with:
      image_name
      annotator ranking orders
      consensus_order
      consensus_votes
      consensus_pct
      decision
      reason
3) Printed colored summary with emojis
4) Optional JSON summary (for reproducibility)

Example:
python rank_consensus_paper.py \
    --jsons "./Annotator3_data/ranking/gt_annotator3.json" "./Annotator3_data/ranking/annotator3_annotator1.json" "./Annotator3_data/ranking/annotator3_annotator2.json" "./Annotator3_data/ranking/annotator3_annotator4.json" \
    --images-dir "./Annotator3_data/ranking/200" \
    --threshold 3 \
    --csv-out ./Annotator3_data/annotator3_consensus_report.csv \
    --summary-json ./Annotator3_data/annotator3_consensus_summary.json

python rank_consensus_paper.py \
    --jsons "./Annotator2_data/ranking/gt_annotator2.json" "./Annotator2_data/ranking/annotator2_annotator1.json" "./Annotator2_data/ranking/annotator2_annotator3.json" "./Annotator2_data/ranking/annotator2_annotator4.json" \
    --images-dir "./Annotator2_data/ranking/100" \
    --threshold 3 \
    --csv-out ./Annotator2_data/annotator2_consensus_report.csv \
    --summary-json ./Annotator2_data/annotator2_consensus_summary.json

python rank_consensus_paper.py \
    --jsons "./Tim_data/ranking/gt_annotator4.json" "./Tim_data/ranking/annotator4_annotator1.json" "./Tim_data/ranking/annotator4_annotator2.json" "./Tim_data/ranking/annotator4_annotator3.json" \
    --images-dir "./Tim_data/ranking/val" \
    --threshold 3 \
    --csv-out ./Tim_data/annotator4_consensus_report.csv \
    --summary-json ./Tim_data/annotator4_consensus_summary.json

python rank_consensus_paper.py \
    --jsons "./annotator1_data/ranking/gt_annotator1.json" "./annotator1_data/ranking/annotator1_annotator3.json" "./annotator1_data/ranking/annotator1_annotator2.json" "./annotator1_data/ranking/annotator1_annotator4.json" \
    --images-dir "./annotator1_data/ranking/test" \
    --threshold 3 \
    --csv-out ./annotator1_data/annotator1_consensus_report.csv \
    --summary-json ./annotator1_data/annotator1_consensus_summary.json

Notes:
- Default behavior is MOVE images.
- Use --copy if you want to preserve the original image folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================
# ANSI colors (summary only)
# ============================================================
class ANSI:
    RESET = "\033[0m"
    BOLD = "\033[1m"

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"


def c(text: str, style: str) -> str:
    return f"{style}{text}{ANSI.RESET}"


def supports_color() -> bool:
    return sys.stdout.isatty()


USE_COLOR = supports_color()


def maybe_color(text: str, style: str) -> str:
    return c(text, style) if USE_COLOR else text


# ============================================================
# Data models
# ============================================================
@dataclass
class AnnotatorData:
    name: str
    json_path: Path
    image_orders: Dict[str, Tuple[int, ...]]  # filename -> ranking order
    num_points_declared: Optional[int] = None
    project: Optional[str] = None


@dataclass
class ImageDecision:
    image_name: str
    annotator_orders: Dict[str, Tuple[int, ...]]  # annotator -> order
    consensus_order: Optional[Tuple[int, ...]]
    consensus_votes: int
    consensus_pct: float
    decision: str  # PASS / FAIL
    reason: str
    file_found: bool = False
    output_path: Optional[str] = None


# ============================================================
# Parsing + validation
# ============================================================
def load_json_file(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load JSON '{path}': {e}") from e


def canonical_order_from_points(points: List[dict], image_name: str, annotator_name: str) -> Tuple[int, ...]:
    """
    Convert a list of point annotations into a canonical ranking order tuple:
    order = tuple(point_id sorted by rank asc)
    """
    if not isinstance(points, list) or len(points) == 0:
        raise ValueError(f"[{annotator_name}] {image_name}: 'points' missing or empty")

    pairs: List[Tuple[int, int]] = []  # (rank, point_id)
    point_ids = set()
    ranks = set()

    for i, p in enumerate(points):
        if not isinstance(p, dict):
            raise ValueError(f"[{annotator_name}] {image_name}: point #{i} is not an object")

        if "point_id" not in p or "rank" not in p:
            raise ValueError(f"[{annotator_name}] {image_name}: point #{i} missing point_id/rank")

        point_id = p["point_id"]
        rank = p["rank"]

        if not isinstance(point_id, int):
            raise ValueError(f"[{annotator_name}] {image_name}: point_id must be int, got {type(point_id).__name__}")
        if rank is None or not isinstance(rank, int):
            raise ValueError(f"[{annotator_name}] {image_name}: rank must be non-null int")

        if point_id in point_ids:
            raise ValueError(f"[{annotator_name}] {image_name}: duplicate point_id={point_id}")
        if rank in ranks:
            raise ValueError(f"[{annotator_name}] {image_name}: duplicate rank={rank}")

        point_ids.add(point_id)
        ranks.add(rank)
        pairs.append((rank, point_id))

    n = len(pairs)
    expected_ranks = set(range(1, n + 1))
    if ranks != expected_ranks:
        raise ValueError(
            f"[{annotator_name}] {image_name}: ranks must be exactly 1..{n}, got {sorted(ranks)}"
        )

    pairs.sort(key=lambda x: x[0])
    return tuple(point_id for _, point_id in pairs)


def load_annotator(json_path: Path, fallback_name: str) -> AnnotatorData:
    data = load_json_file(json_path)

    ranker = str(data.get("ranker") or fallback_name)
    project = data.get("project")
    num_points_declared = data.get("numPoints")

    images = data.get("images")
    if not isinstance(images, list):
        raise ValueError(f"[{ranker}] JSON '{json_path}' missing valid 'images' list")

    image_orders: Dict[str, Tuple[int, ...]] = {}

    for img in images:
        if not isinstance(img, dict):
            raise ValueError(f"[{ranker}] Invalid image record in '{json_path}'")

        filename = img.get("filename")
        points = img.get("points")
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(f"[{ranker}] Invalid/missing image filename in '{json_path}'")

        order = canonical_order_from_points(points, filename, ranker)
        image_orders[filename] = order

    return AnnotatorData(
        name=ranker,
        json_path=json_path,
        image_orders=image_orders,
        num_points_declared=num_points_declared if isinstance(num_points_declared, int) else None,
        project=str(project) if project is not None else None,
    )


def ensure_unique_names(annotators: List[AnnotatorData]) -> None:
    """
    Make annotator names unique for CSV columns.
    If duplicates exist, append suffixes: name, name_2, name_3...
    """
    seen = Counter()
    for ann in annotators:
        seen[ann.name] += 1
        if seen[ann.name] > 1:
            ann.name = f"{ann.name}_{seen[ann.name]}"


def validate_cross_annotator_consistency(annotators: List[AnnotatorData], strict_image_set: bool) -> List[str]:
    """
    Returns warnings. Raises only for fatal issues.
    """
    warnings: List[str] = []

    if not annotators:
        raise ValueError("No annotators loaded")

    # Check numPoints consistency (if available)
    declared_points = [a.num_points_declared for a in annotators if a.num_points_declared is not None]
    if declared_points:
        if len(set(declared_points)) > 1:
            warnings.append(f"numPoints differs across JSONs: {declared_points}")

    # Check project consistency (if available)
    projects = [a.project for a in annotators if a.project]
    if projects and len(set(projects)) > 1:
        warnings.append(f"project name differs across JSONs: {sorted(set(projects))}")

    # Check image set consistency
    image_sets = [set(a.image_orders.keys()) for a in annotators]
    union = set().union(*image_sets)
    inter = set.intersection(*image_sets) if image_sets else set()

    if union != inter:
        missing_report = []
        for ann in annotators:
            missing = sorted(union - set(ann.image_orders.keys()))
            if missing:
                missing_report.append(f"{ann.name} missing {len(missing)} image(s)")
        msg = "Image set mismatch across annotators: " + "; ".join(missing_report)
        if strict_image_set:
            raise ValueError(msg)
        warnings.append(msg)

    # Check order tuple lengths per image across annotators
    all_images = sorted(union)
    for image_name in all_images:
        lengths = {}
        for ann in annotators:
            if image_name in ann.image_orders:
                lengths[ann.name] = len(ann.image_orders[image_name])
        if len(set(lengths.values())) > 1:
            raise ValueError(
                f"Point count mismatch for image '{image_name}' across annotators: {lengths}"
            )

    return warnings


# ============================================================
# Consensus
# ============================================================
def compute_consensus(
    annotators: List[AnnotatorData],
    threshold: int,
) -> List[ImageDecision]:
    all_images = sorted(set().union(*(a.image_orders.keys() for a in annotators)))
    decisions: List[ImageDecision] = []

    n_annotators = len(annotators)

    for image_name in all_images:
        annotator_orders: Dict[str, Tuple[int, ...]] = {}
        missing = []

        for ann in annotators:
            order = ann.image_orders.get(image_name)
            if order is None:
                missing.append(ann.name)
            else:
                annotator_orders[ann.name] = order

        # Missing annotation => FAIL (keeps output strictly pass/fail as requested)
        if missing:
            decisions.append(
                ImageDecision(
                    image_name=image_name,
                    annotator_orders=annotator_orders,
                    consensus_order=None,
                    consensus_votes=0,
                    consensus_pct=0.0,
                    decision="FAIL",
                    reason=f"Missing annotation(s): {', '.join(missing)}",
                )
            )
            continue

        vote_counter = Counter(annotator_orders.values())
        consensus_order, consensus_votes = vote_counter.most_common(1)[0]
        consensus_pct = 100.0 * consensus_votes / n_annotators
        decision = "PASS" if consensus_votes >= threshold else "FAIL"

        # Optional richer reason in case of ties/conflicts
        if decision == "PASS":
            reason = f"Exact-order consensus {consensus_votes}/{n_annotators}"
        else:
            unique_orders = len(vote_counter)
            reason = f"No majority ({consensus_votes}/{n_annotators}), {unique_orders} unique order(s)"

        decisions.append(
            ImageDecision(
                image_name=image_name,
                annotator_orders=annotator_orders,
                consensus_order=consensus_order,
                consensus_votes=consensus_votes,
                consensus_pct=consensus_pct,
                decision=decision,
                reason=reason,
            )
        )

    return decisions


# ============================================================
# File operations
# ============================================================
def find_image_source(images_dir: Path, image_name: str) -> Optional[Path]:
    """
    Find image in:
      images_dir/image_name
      images_dir/pass/image_name
      images_dir/fail/image_name
    """
    candidates = [
        images_dir / image_name,
        images_dir / "pass" / image_name,
        images_dir / "fail" / image_name,
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def place_images(
    decisions: List[ImageDecision],
    images_dir: Path,
    copy_mode: bool,
    dry_run: bool,
) -> None:
    pass_dir = images_dir / "pass"
    fail_dir = images_dir / "fail"

    if not dry_run:
        pass_dir.mkdir(parents=True, exist_ok=True)
        fail_dir.mkdir(parents=True, exist_ok=True)

    for d in decisions:
        src = find_image_source(images_dir, d.image_name)
        d.file_found = src is not None

        target_dir = pass_dir if d.decision == "PASS" else fail_dir
        dst = target_dir / d.image_name
        d.output_path = str(dst)

        if src is None:
            d.reason += " | image file not found"
            continue

        if dry_run:
            continue

        # If source and destination are same path, nothing to do
        try:
            if src.resolve() == dst.resolve():
                continue
        except FileNotFoundError:
            # just continue with copy/move
            pass

        # Remove existing destination to avoid stale duplicates
        if dst.exists():
            dst.unlink()

        if copy_mode:
            shutil.copy2(src, dst)
        else:
            shutil.move(str(src), str(dst))


# ============================================================
# CSV + JSON outputs
# ============================================================
def order_to_str(order: Optional[Tuple[int, ...]]) -> str:
    if order is None:
        return ""
    return ">".join(map(str, order))


def write_csv_report(
    decisions: List[ImageDecision],
    annotators: List[AnnotatorData],
    csv_path: Path,
) -> None:
    annotator_names = [a.name for a in annotators]

    # Paper-friendly columns:
    headers = [
        "image_name",
        "n_annotators_expected",
        "n_annotators_present",
    ]
    headers += [f"{name}_order" for name in annotator_names]
    headers += [f"{name}_matches_consensus" for name in annotator_names]
    headers += [
        "consensus_order",
        "consensus_votes",
        "consensus_pct",
        "decision",
        "reason",
        "file_found",
        "output_path",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for d in decisions:
            row = [
                d.image_name,
                len(annotators),
                len(d.annotator_orders),
            ]

            # annotator orders
            for ann in annotator_names:
                row.append(order_to_str(d.annotator_orders.get(ann)))

            # per-annotator consensus flags
            for ann in annotator_names:
                order = d.annotator_orders.get(ann)
                if order is None or d.consensus_order is None:
                    row.append("")
                else:
                    row.append("1" if order == d.consensus_order else "0")

            row += [
                order_to_str(d.consensus_order),
                d.consensus_votes,
                f"{d.consensus_pct:.2f}",
                d.decision,
                d.reason,
                "1" if d.file_found else "0",
                d.output_path or "",
            ]
            writer.writerow(row)


def write_summary_json(
    summary_json_path: Path,
    decisions: List[ImageDecision],
    annotators: List[AnnotatorData],
    threshold: int,
    copy_mode: bool,
    dry_run: bool,
) -> None:
    # Compact but reproducible structure
    payload = {
        "method": "exact_order_majority_consensus",
        "threshold": threshold,
        "n_annotators": len(annotators),
        "copy_mode": copy_mode,
        "dry_run": dry_run,
        "annotators": [
            {
                "name": a.name,
                "json_path": str(a.json_path),
                "num_images": len(a.image_orders),
                "numPoints_declared": a.num_points_declared,
                "project": a.project,
            }
            for a in annotators
        ],
        "decisions": [
            {
                "image_name": d.image_name,
                "annotator_orders": {k: list(v) for k, v in d.annotator_orders.items()},
                "consensus_order": list(d.consensus_order) if d.consensus_order else None,
                "consensus_votes": d.consensus_votes,
                "consensus_pct": d.consensus_pct,
                "decision": d.decision,
                "reason": d.reason,
                "file_found": d.file_found,
                "output_path": d.output_path,
            }
            for d in decisions
        ],
    }

    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ============================================================
# Metrics + summary
# ============================================================
def summarize_annotators(decisions: List[ImageDecision], annotators: List[AnnotatorData]) -> List[dict]:
    """
    Returns paper-friendly per-annotator metrics:
    - overall_consensus_agreement_pct: matches consensus on all images with valid consensus
    - pass_consensus_agreement_pct: matches consensus restricted to PASS images
    - pass_yield_pct: % of all images where annotator's order is the winning consensus AND image passed
      (this aligns well with "each json percentage that yield pass based on his ranking order")
    """
    total_images = len(decisions)
    rows = []

    for ann in annotators:
        overall_total = 0
        overall_match = 0

        pass_total = 0
        pass_match = 0

        pass_yield_count = 0  # among all images, contributed the consensus order of a PASS image

        for d in decisions:
            ann_order = d.annotator_orders.get(ann.name)
            if ann_order is None or d.consensus_order is None:
                continue

            overall_total += 1
            if ann_order == d.consensus_order:
                overall_match += 1

            if d.decision == "PASS":
                pass_total += 1
                if ann_order == d.consensus_order:
                    pass_match += 1
                    pass_yield_count += 1

        overall_pct = (100.0 * overall_match / overall_total) if overall_total else 0.0
        pass_agree_pct = (100.0 * pass_match / pass_total) if pass_total else 0.0
        pass_yield_pct = (100.0 * pass_yield_count / total_images) if total_images else 0.0

        rows.append(
            {
                "annotator": ann.name,
                "overall_match": overall_match,
                "overall_total": overall_total,
                "overall_pct": overall_pct,
                "pass_match": pass_match,
                "pass_total": pass_total,
                "pass_agree_pct": pass_agree_pct,
                "pass_yield_count": pass_yield_count,
                "pass_yield_pct": pass_yield_pct,
            }
        )

    return rows


def print_summary(
    decisions: List[ImageDecision],
    annotators: List[AnnotatorData],
    threshold: int,
    csv_path: Path,
    warnings: List[str],
    copy_mode: bool,
    dry_run: bool,
) -> None:
    total = len(decisions)
    passed = sum(1 for d in decisions if d.decision == "PASS")
    failed = total - passed
    file_missing = sum(1 for d in decisions if not d.file_found)

    print()
    print(maybe_color("════════════════════════════════════════════════════════════", ANSI.CYAN))
    print(maybe_color("📘 Paper-Ready Consensus Summary", ANSI.BOLD + ANSI.CYAN))
    print(maybe_color("════════════════════════════════════════════════════════════", ANSI.CYAN))

    action_word = "COPY" if copy_mode else "MOVE"
    mode_text = f"{action_word} mode" + (" (dry-run)" if dry_run else "")
    print(f"{maybe_color('⚙️  Mode:', ANSI.BLUE)} {mode_text}")
    print(f"{maybe_color('🧑‍🤝‍🧑 Annotators:', ANSI.MAGENTA)} {len(annotators)}")
    print(f"{maybe_color('🎯 PASS rule:', ANSI.YELLOW)} exact-order majority ≥ {threshold}/{len(annotators)}")
    print(f"{maybe_color('📝 CSV:', ANSI.CYAN)} {csv_path}")

    print()
    print(
        f"{maybe_color('✅ PASS', ANSI.GREEN)}: {passed}   "
        f"{maybe_color('❌ FAIL', ANSI.RED)}: {failed}   "
        f"{maybe_color('🧮 TOTAL', ANSI.MAGENTA)}: {total}"
    )

    if total > 0:
        pass_pct = 100.0 * passed / total
        fail_pct = 100.0 * failed / total
        print(
            f"{maybe_color('📈 PASS %', ANSI.GREEN)}: {pass_pct:.2f}%   "
            f"{maybe_color('📉 FAIL %', ANSI.RED)}: {fail_pct:.2f}%"
        )

    if file_missing > 0:
        print(maybe_color(f"⚠️  Missing image files: {file_missing}", ANSI.YELLOW))

    if warnings:
        print()
        print(maybe_color("⚠️  Warnings", ANSI.BOLD + ANSI.YELLOW))
        for w in warnings:
            print(f"  • {w}")

    # Failure reason breakdown
    fail_reasons = Counter(d.reason.split(" | ")[0] for d in decisions if d.decision == "FAIL")
    if fail_reasons:
        print()
        print(maybe_color("🧩 Failure breakdown", ANSI.BOLD + ANSI.YELLOW))
        for reason, cnt in fail_reasons.most_common():
            print(f"  • {reason}: {cnt}")

    # Per-annotator metrics
    metrics = summarize_annotators(decisions, annotators)
    print()
    print(maybe_color("👥 Per-annotator agreement (paper metrics)", ANSI.BOLD + ANSI.CYAN))
    print("  " + "-" * 86)
    for m in metrics:
        # color by PASS agreement
        if m["pass_agree_pct"] >= 90:
            emoji = "🟢"
            style = ANSI.GREEN
        elif m["pass_agree_pct"] >= 75:
            emoji = "🟡"
            style = ANSI.YELLOW
        else:
            emoji = "🔴"
            style = ANSI.RED

        annotator_label = maybe_color(m["annotator"], ANSI.BOLD)
        pass_agree_text = maybe_color(
            f"{m['pass_match']}/{m['pass_total']} ({m['pass_agree_pct']:.2f}%)", style
        )
        overall_text = maybe_color(
            f"{m['overall_match']}/{m['overall_total']} ({m['overall_pct']:.2f}%)", ANSI.GRAY
        )
        yield_text = maybe_color(
            f"{m['pass_yield_count']}/{total} ({m['pass_yield_pct']:.2f}%)", ANSI.CYAN
        )

        print(
            f"  {emoji} {annotator_label}\n"
            f"     • PASS-consensus agreement : {pass_agree_text}\n"
            f"     • Overall consensus agree  : {overall_text}\n"
            f"     • PASS yield contribution  : {yield_text}"
        )

    print(maybe_color("════════════════════════════════════════════════════════════", ANSI.CYAN))
    print()


# ============================================================
# CLI
# ============================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Consensus split (PASS/FAIL) for depth ranking annotations (paper-ready reporting)."
    )
    p.add_argument(
        "--jsons",
        nargs="+",
        required=True,
        help="List of ranking JSON files (e.g., 4 annotators).",
    )
    p.add_argument(
        "--images-dir",
        required=True,
        help="Folder containing the image files referenced by the JSONs.",
    )
    p.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="Consensus threshold (e.g., 3 for 3/4). Default = ceil(N/2).",
    )
    p.add_argument(
        "--csv-out",
        default="consensus_report.csv",
        help="Output CSV report path.",
    )
    p.add_argument(
        "--summary-json",
        default=None,
        help="Optional JSON summary path (recommended for reproducibility).",
    )
    p.add_argument(
        "--copy",
        action="store_true",
        help="Copy images into pass/fail instead of moving them.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute everything but do not move/copy files.",
    )
    p.add_argument(
        "--strict-image-set",
        action="store_true",
        help="Fail if JSONs do not contain exactly the same image set.",
    )
    return p


def validate_args(args: argparse.Namespace, n_annotators: int) -> None:
    if n_annotators <= 0:
        raise ValueError("No JSONs provided")

    if args.threshold is None:
        args.threshold = math.ceil(n_annotators / 2)

    if not (1 <= args.threshold <= n_annotators):
        raise ValueError(
            f"--threshold must be in [1, {n_annotators}], got {args.threshold}"
        )


# ============================================================
# Main
# ============================================================
def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    json_paths = [Path(p).expanduser().resolve() for p in args.jsons]
    images_dir = Path(args.images_dir).expanduser().resolve()
    csv_out = Path(args.csv_out).expanduser().resolve()
    summary_json = Path(args.summary_json).expanduser().resolve() if args.summary_json else None

    # Path checks
    for jp in json_paths:
        if not jp.exists():
            print(maybe_color(f"❌ JSON not found: {jp}", ANSI.RED), file=sys.stderr)
            return 1
        if not jp.is_file():
            print(maybe_color(f"❌ Not a file: {jp}", ANSI.RED), file=sys.stderr)
            return 1

    if not images_dir.exists() or not images_dir.is_dir():
        print(maybe_color(f"❌ Invalid --images-dir: {images_dir}", ANSI.RED), file=sys.stderr)
        return 1

    try:
        annotators: List[AnnotatorData] = []
        for i, jp in enumerate(json_paths, start=1):
            annotators.append(load_annotator(jp, fallback_name=f"ranker_{i}"))

        ensure_unique_names(annotators)
        validate_args(args, n_annotators=len(annotators))
        warnings = validate_cross_annotator_consistency(annotators, strict_image_set=args.strict_image_set)

        decisions = compute_consensus(annotators, threshold=args.threshold)

        # File placement (move/copy)
        place_images(
            decisions=decisions,
            images_dir=images_dir,
            copy_mode=args.copy,
            dry_run=args.dry_run,
        )

        # CSV report
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        write_csv_report(decisions, annotators, csv_out)

        # Optional JSON summary
        if summary_json:
            summary_json.parent.mkdir(parents=True, exist_ok=True)
            write_summary_json(
                summary_json_path=summary_json,
                decisions=decisions,
                annotators=annotators,
                threshold=args.threshold,
                copy_mode=args.copy,
                dry_run=args.dry_run,
            )

        print_summary(
            decisions=decisions,
            annotators=annotators,
            threshold=args.threshold,
            csv_path=csv_out,
            warnings=warnings,
            copy_mode=args.copy,
            dry_run=args.dry_run,
        )

        if summary_json:
            print(f"{maybe_color('📦 Summary JSON:', ANSI.CYAN)} {summary_json}")

        return 0

    except Exception as e:
        print(maybe_color(f"❌ Error: {e}", ANSI.RED), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())