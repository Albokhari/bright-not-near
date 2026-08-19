#!/usr/bin/env python3
"""
build_missing_ranker_json.py

Reconstruct a missing ranker annotation JSON using:
  1) A GT annotation JSON for point coordinates (x,y) and point_ids
  2) A ranking table (CSV-like) containing per-image ranking orders for a given ranker column

This version is robust to "CSV" files where list-valued columns are NOT quoted, e.g.:
  ..., [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], ...

It splits commas only when outside brackets.

Example:
  python build_missing_ranker_json.py \
    --ranking-csv agreement_levels.csv \
    --gt-json gt_annotator1.json \
    --ranking-column ranker_1_ranking \
    --ranker-name Annotator4 \
    --out-json annotator1_annotator4.json \
    --strict
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


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


def parse_order_list(s: str, *, filename: str, col: str) -> List[int]:
    """Parse strings like '[1, 2, 3, 4, 5]' into list[int]."""
    if s is None:
        raise ValueError(f"{filename}: missing value in column '{col}'")

    s = str(s).strip()
    if not s:
        raise ValueError(f"{filename}: empty value in column '{col}'")

    try:
        v = ast.literal_eval(s)
    except Exception as e:
        raise ValueError(f"{filename}: cannot parse '{col}' as list: {s!r} ({e})") from e

    if not isinstance(v, list) or not all(isinstance(x, int) for x in v):
        raise ValueError(f"{filename}: '{col}' must be a list of ints, got: {v!r}")

    if len(v) == 0:
        raise ValueError(f"{filename}: '{col}' list is empty")

    if len(set(v)) != len(v):
        raise ValueError(f"{filename}: '{col}' has duplicates: {v!r}")

    return v


def split_commas_outside_brackets(line: str) -> List[str]:
    """
    Split a line on commas, but ignore commas inside:
      - square brackets [...]
      - double quotes "..."
      - single quotes '...'

    This is designed for your "CSV-like" file where list columns are unquoted.
    """
    parts: List[str] = []
    buf: List[str] = []

    bracket_depth = 0
    in_single_quote = False
    in_double_quote = False
    escape = False

    for ch in line:
        if escape:
            buf.append(ch)
            escape = False
            continue

        if ch == "\\" and (in_single_quote or in_double_quote):
            buf.append(ch)
            escape = True
            continue

        # quote state
        if ch == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            buf.append(ch)
            continue

        if ch == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            buf.append(ch)
            continue

        # bracket depth only matters when not inside quotes
        if not in_single_quote and not in_double_quote:
            if ch == "[":
                bracket_depth += 1
                buf.append(ch)
                continue
            elif ch == "]":
                bracket_depth = max(0, bracket_depth - 1)
                buf.append(ch)
                continue
            elif ch == "," and bracket_depth == 0:
                parts.append("".join(buf).strip())
                buf = []
                continue

        buf.append(ch)

    parts.append("".join(buf).strip())
    return parts


def read_rankings_csv_flexible(
    csv_path: Path,
    ranking_column: str,
    filename_column: str = "filename",
) -> Dict[str, List[int]]:
    """
    Reads a "CSV-like" table robustly, including unquoted list columns.
    Returns: filename -> order list
    """
    try:
        with csv_path.open("r", encoding="utf-8") as f:
            raw_lines = [ln.rstrip("\n").rstrip("\r") for ln in f if ln.strip()]

        if not raw_lines:
            raise ValueError("File is empty")

        header_line = raw_lines[0]
        headers = [h.strip() for h in header_line.split(",")]

        if filename_column not in headers:
            raise ValueError(f"Missing filename column '{filename_column}'. Found: {headers}")
        if ranking_column not in headers:
            raise ValueError(f"Missing ranking column '{ranking_column}'. Found: {headers}")

        expected_ncols = len(headers)
        filename_idx = headers.index(filename_column)
        ranking_idx = headers.index(ranking_column)

        mapping: Dict[str, List[int]] = {}

        for line_no, line in enumerate(raw_lines[1:], start=2):
            cols = split_commas_outside_brackets(line)

            if len(cols) != expected_ncols:
                raise ValueError(
                    f"Line {line_no}: expected {expected_ncols} columns, got {len(cols)}.\n"
                    f"Header: {headers}\n"
                    f"Parsed: {cols}"
                )

            fn = cols[filename_idx].strip()
            if not fn:
                continue

            if fn in mapping:
                raise ValueError(f"Duplicate filename row in table: {fn} (line {line_no})")

            order_str = cols[ranking_idx]
            order = parse_order_list(order_str, filename=fn, col=ranking_column)
            mapping[fn] = order

        if not mapping:
            raise ValueError("No rows parsed from ranking table")

        return mapping

    except Exception as e:
        raise RuntimeError(f"Failed reading CSV: {csv_path} ({e})") from e


def build_ranker_json_from_gt(
    gt: dict,
    orders_by_filename: Dict[str, List[int]],
    ranker_name: str,
    ranking_column: str,
    created_at: str,
    strict: bool,
) -> dict:
    if not isinstance(gt, dict):
        raise ValueError("GT JSON root must be an object")

    gt_images = gt.get("images")
    if not isinstance(gt_images, list):
        raise ValueError("GT JSON missing 'images' list")

    project = gt.get("project")
    num_points = gt.get("numPoints")

    if not isinstance(num_points, int) or num_points <= 0:
        raise ValueError(f"GT JSON has invalid numPoints: {num_points!r}")

    out_images: List[dict] = []
    used_filenames = set()

    for img in gt_images:
        if not isinstance(img, dict):
            raise ValueError("GT JSON images entries must be objects")

        fn = img.get("filename")
        if not isinstance(fn, str) or not fn.strip():
            raise ValueError("GT JSON image missing valid filename")
        fn = fn.strip()

        if fn not in orders_by_filename:
            if strict:
                raise ValueError(f"Missing ranking for GT image in CSV: {fn}")
            continue

        order = orders_by_filename[fn]
        used_filenames.add(fn)

        if len(order) != num_points:
            raise ValueError(
                f"{fn}: ranking length {len(order)} != GT numPoints {num_points}. order={order}"
            )

        points = img.get("points")
        if not isinstance(points, list) or len(points) != num_points:
            raise ValueError(f"{fn}: GT points list invalid or length mismatch")

        gt_point_ids: List[int] = []
        point_lookup: Dict[int, dict] = {}
        for p in points:
            if not isinstance(p, dict):
                raise ValueError(f"{fn}: GT point entry not an object")
            pid = p.get("point_id")
            if not isinstance(pid, int):
                raise ValueError(f"{fn}: GT point_id must be int, got {pid!r}")
            if pid in point_lookup:
                raise ValueError(f"{fn}: duplicate GT point_id: {pid}")
            gt_point_ids.append(pid)
            point_lookup[pid] = p

        gt_set = set(gt_point_ids)
        order_set = set(order)
        if order_set != gt_set:
            missing = sorted(gt_set - order_set)
            extra = sorted(order_set - gt_set)
            raise ValueError(
                f"{fn}: ranking point_ids do not match GT point_ids.\n"
                f"  missing_from_ranking={missing}\n"
                f"  extra_in_ranking={extra}\n"
                f"  gt_point_ids={sorted(gt_set)}\n"
                f"  ranking_order={order}"
            )

        # Preserve original GT point order, only rewrite rank values
        new_points = [dict(point_lookup[pid]) for pid in gt_point_ids]

        rank_by_pid = {pid: i + 1 for i, pid in enumerate(order)}
        for p in new_points:
            p["rank"] = rank_by_pid[p["point_id"]]

        new_img = dict(img)
        new_img["points"] = new_points
        out_images.append(new_img)

    unused = sorted(set(orders_by_filename.keys()) - used_filenames)
    if unused:
        msg = (
            f"CSV contains {len(unused)} filename(s) not present in GT JSON. "
            f"Examples: {unused[:5]}"
        )
        if strict:
            raise ValueError(msg)
        else:
            print(f"⚠️ Warning: {msg}", file=sys.stderr)

    out = {
        "type": "depth_annotation_ranking",
        "ranker": ranker_name,
        "sourceRankingColumn": ranking_column,
        "project": project,
        "numPoints": num_points,
        "createdAt": created_at,
        "images": out_images,
    }
    return out


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reconstruct a missing ranker JSON using GT coordinates + ranking orders."
    )
    p.add_argument("--ranking-csv", required=True, help="CSV-like file containing ranking orders")
    p.add_argument("--gt-json", required=True, help="GT annotation JSON with point coordinates")
    p.add_argument("--ranking-column", required=True, help="Column name, e.g. ranker_1_ranking")
    p.add_argument("--filename-column", default="filename", help="Filename column name")
    p.add_argument("--ranker-name", default="ranker_1", help="Output 'ranker' field value")
    p.add_argument("--out-json", default="ranker_1_ranking.json", help="Output JSON file")
    p.add_argument("--created-at", default=None, help="Optional ISO-8601 timestamp")
    p.add_argument("--strict", action="store_true", help="Require exact CSV/GT filename match")
    return p


def main() -> int:
    args = build_parser().parse_args()

    csv_path = Path(args.ranking_csv).expanduser().resolve()
    gt_path = Path(args.gt_json).expanduser().resolve()
    out_path = Path(args.out_json).expanduser().resolve()

    if not csv_path.exists():
        print(f"❌ CSV not found: {csv_path}", file=sys.stderr)
        return 1
    if not gt_path.exists():
        print(f"❌ GT JSON not found: {gt_path}", file=sys.stderr)
        return 1

    created_at = args.created_at or utc_now_iso()

    try:
        orders = read_rankings_csv_flexible(
            csv_path=csv_path,
            ranking_column=args.ranking_column,
            filename_column=args.filename_column,
        )
        gt = load_json(gt_path)

        out = build_ranker_json_from_gt(
            gt=gt,
            orders_by_filename=orders,
            ranker_name=args.ranker_name,
            ranking_column=args.ranking_column,
            created_at=created_at,
            strict=args.strict,
        )

        save_json(out_path, out)

        print("✅ Reconstructed ranker JSON written:")
        print(f"   {out_path}")
        print()
        print("Summary:")
        print(f"  ranker         : {out['ranker']}")
        print(f"  project        : {out.get('project')}")
        print(f"  numPoints      : {out.get('numPoints')}")
        print(f"  images_written : {len(out.get('images', []))}")
        print(f"  createdAt      : {out.get('createdAt')}")
        return 0

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())