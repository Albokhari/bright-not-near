#!/usr/bin/env python3
"""
UPDATED FULL SCRIPT (works with your file)

Your situation (based on your traceback):
- Your file is a NORMAL comma-separated CSV on disk, like:
    filename,agreement_level,total_inversions,...,[1, 2, 3, 4, 5],...

But you ALSO previously pasted a tab-separated view where rankings look split.
This script handles BOTH:

It reads line-by-line and:
- auto-detects delimiter per line (comma vs tab, whichever appears more)
- reconstructs each ranker ordering by consuming 5 integers per ranker, even if
  the ranking is split across multiple tokens/cells.

Rule:
- PASS if >= 3 rankers have exactly the same full ordering (the 5-number sequence).
- For PASS rows, keep_annotation_from is chosen from agreeing rankers using preference order.

Outputs (default: same directory as input):
- ranking_processed.csv
- pass_list.txt
- review_list.txt
- keep_annotation_map.csv

Run (Windows):
  python process_rankings.py --input "C:\mnt\agreement_levels.csv"
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


# ------------------------- constants & regex -------------------------

INT_RE = re.compile(r"-?\d+")

EXPECTED_COLS = [
    "filename",
    "agreement_level",
    "total_inversions",
    "unique_rankings",
    "max_rankers_agreeing",
    "annotator1_ranking",
    "ranker_1_ranking",
    "annotator2_ranking",
    "annotator3_ranking",
]


# ------------------------- small helpers -------------------------

def ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def write_list(path: str, items: Iterable[str]) -> None:
    ensure_dir_for_file(path)
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(f"{x}\n")


def parse_ints_from_text(s: str) -> List[int]:
    return [int(x) for x in INT_RE.findall(s or "")]


def consume_ranking(tokens: List[str], start_idx: int, expected_len: int = 5) -> Tuple[Tuple[int, ...], int]:
    """
    Consume tokens starting at start_idx until we collect expected_len integers.
    Works for:
      - CSV one token: ["[1, 2, 3, 4, 5]"]
      - TSV split tokens: ["[1", "2", "3", "4", " 5]"]
      - Any mix of weird whitespace/brackets/commas
    Returns: (ranking_tuple, next_idx)
    """
    ints: List[int] = []
    idx = start_idx

    while idx < len(tokens) and len(ints) < expected_len:
        ints.extend(parse_ints_from_text(tokens[idx]))
        idx += 1

    if len(ints) < expected_len:
        raise ValueError(
            f"Could not parse {expected_len} ints for ranking starting at token {start_idx}. "
            f"Got {len(ints)} ints. Tokens slice: {tokens[start_idx:start_idx+30]}"
        )

    return tuple(ints[:expected_len]), idx


def guess_delimiter(line: str) -> str:
    """
    Decide delimiter for a given line. We choose the one that appears more.
    If equal or neither, default to comma.
    """
    c = line.count(",")
    t = line.count("\t")
    if t > c:
        return "\t"
    return ","


# ------------------------- reading your file -------------------------

def read_agreement_levels_flexible(path: str) -> pd.DataFrame:
    """
    Reads either:
      A) Normal comma-separated CSV:
         filename,agreement_level,..., [1, 2, 3, 4, 5], ...
      B) Tab-separated where rankings may be split across cells.

    We DO NOT rely on pandas csv parsing because the "split ranking cells" case breaks it.
    Instead we parse each line ourselves and reconstruct the 4 rankings by consuming 5 integers each.
    """
    rows: List[Dict[str, str]] = []

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        header_line = f.readline()
        if not header_line:
            raise ValueError("Empty file")

        # We don't trust header columns to align (extra tabs etc).
        # But we can sanity-check it:
        if "filename" not in header_line.lower():
            raise ValueError("Header does not look right: expected 'filename' in first line.")

        for lineno, raw in enumerate(f, start=2):
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue

            delim = guess_delimiter(line)
            tokens = line.split(delim)

            # Trim trailing empties
            while tokens and tokens[-1] == "":
                tokens.pop()

            if len(tokens) < 5:
                raise ValueError(
                    f"Line {lineno}: too few columns (<5) after splitting by {delim!r}. Line: {line}"
                )

            filename = tokens[0].strip()
            agreement_level = tokens[1].strip()
            total_inversions = tokens[2].strip()
            unique_rankings = tokens[3].strip()
            max_rankers_agreeing = tokens[4].strip()

            idx = 5
            annotator1, idx = consume_ranking(tokens, idx, expected_len=5)
            ranker1, idx = consume_ranking(tokens, idx, expected_len=5)
            annotator2, idx = consume_ranking(tokens, idx, expected_len=5)
            annotator3, idx = consume_ranking(tokens, idx, expected_len=5)

            rows.append(
                {
                    "filename": filename,
                    "agreement_level": agreement_level,
                    "total_inversions": total_inversions,
                    "unique_rankings": unique_rankings,
                    "max_rankers_agreeing": max_rankers_agreeing,
                    "annotator1_ranking": " ".join(map(str, annotator1)),
                    "ranker_1_ranking": " ".join(map(str, ranker1)),
                    "annotator2_ranking": " ".join(map(str, annotator2)),
                    "annotator3_ranking": " ".join(map(str, annotator3)),
                }
            )

    df = pd.DataFrame(rows, columns=EXPECTED_COLS)

    # numeric conversions (optional but useful)
    for c in ["total_inversions", "unique_rankings", "max_rankers_agreeing"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# ------------------------- consensus logic -------------------------

@dataclass
class RowDecision:
    status: str  # PASS / REVIEW
    agree_count: int
    consensus_ordering: Optional[Tuple[int, ...]]
    matching_rankers: List[str]
    keep_annotation_from: str
    reason: str


def parse_ranking_str_to_tuple(s: str) -> Optional[Tuple[int, ...]]:
    s = (s or "").strip()
    if not s:
        return None
    nums = [int(x) for x in INT_RE.findall(s)]
    return tuple(nums) if nums else None


def decide_for_row(
    row: pd.Series,
    ranker_cols: Sequence[str],
    min_agree: int,
    source_preference: Sequence[str],
) -> RowDecision:
    parsed: List[Tuple[int, ...]] = []
    parsed_by_col: Dict[str, Tuple[int, ...]] = {}

    for col in ranker_cols:
        t = parse_ranking_str_to_tuple(str(row.get(col, "")))
        if t is not None:
            parsed.append(t)
            parsed_by_col[col] = t

    if not parsed:
        return RowDecision(
            status="REVIEW",
            agree_count=0,
            consensus_ordering=None,
            matching_rankers=[],
            keep_annotation_from="",
            reason="no_rankings",
        )

    counts = Counter(parsed)
    consensus_ordering, agree_count = counts.most_common(1)[0]
    matching = [col for col, t in parsed_by_col.items() if t == consensus_ordering]

    if agree_count >= min_agree:
        keep_from = ""
        for pref in source_preference:
            if pref in matching:
                keep_from = pref
                break
        if not keep_from and matching:
            keep_from = matching[0]

        return RowDecision(
            status="PASS",
            agree_count=int(agree_count),
            consensus_ordering=consensus_ordering,
            matching_rankers=matching,
            keep_annotation_from=keep_from,
            reason=f"{agree_count}_identical_rankings",
        )

    return RowDecision(
        status="REVIEW",
        agree_count=int(agree_count),
        consensus_ordering=consensus_ordering,
        matching_rankers=matching,
        keep_annotation_from="",
        reason=f"max_identical={agree_count}_below_{min_agree}",
    )


def ordering_to_str(ordering: Optional[Tuple[int, ...]]) -> str:
    if ordering is None:
        return ""
    return " ".join(str(x) for x in ordering)


# ------------------------- CLI -------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Process agreement_levels file and select PASS/REVIEW + which annotation to keep.")
    p.add_argument("--input", "-i", required=True, help='Input path, e.g. "C:\\mnt\\agreement_levels.csv"')
    p.add_argument("--min-agree", type=int, default=3, help="PASS if >= this many identical full rankings. Default: 3")

    p.add_argument(
        "--ranker-cols",
        nargs="+",
        default=["annotator1_ranking", "ranker_1_ranking", "annotator2_ranking", "annotator3_ranking"],
        help="Ranking columns to compare.",
    )
    p.add_argument(
        "--source-preference",
        nargs="+",
        default=["annotator1_ranking", "ranker_1_ranking", "annotator2_ranking", "annotator3_ranking"],
        help="When PASS, keep annotation from first matching ranker in this order.",
    )

    p.add_argument("--output", default="", help="Output CSV path. Default: <input_dir>\\ranking_processed.csv")
    p.add_argument("--pass-list", default="", help="PASS list path. Default: <input_dir>\\pass_list.txt")
    p.add_argument("--review-list", default="", help="REVIEW list path. Default: <input_dir>\\review_list.txt")
    p.add_argument("--keep-map", default="", help="Keep map path. Default: <input_dir>\\keep_annotation_map.csv")
    return p


def main() -> None:
    args = build_parser().parse_args()

    input_dir = os.path.dirname(os.path.abspath(args.input))
    output_csv = args.output or os.path.join(input_dir, "ranking_processed.csv")
    pass_list_path = args.pass_list or os.path.join(input_dir, "pass_list.txt")
    review_list_path = args.review_list or os.path.join(input_dir, "review_list.txt")
    keep_map_path = args.keep_map or os.path.join(input_dir, "keep_annotation_map.csv")

    df = read_agreement_levels_flexible(args.input)

    statuses: List[str] = []
    agree_counts: List[int] = []
    consensus_strs: List[str] = []
    matching_rankers_strs: List[str] = []
    keep_from_list: List[str] = []
    reasons: List[str] = []

    for _, row in df.iterrows():
        d = decide_for_row(
            row=row,
            ranker_cols=args.ranker_cols,
            min_agree=args.min_agree,
            source_preference=args.source_preference,
        )
        statuses.append(d.status)
        agree_counts.append(d.agree_count)
        consensus_strs.append(ordering_to_str(d.consensus_ordering))
        matching_rankers_strs.append(",".join(d.matching_rankers))
        keep_from_list.append(d.keep_annotation_from)
        reasons.append(d.reason)

    out = df.copy()
    out["status"] = statuses
    out["agree_count"] = agree_counts
    out["consensus_ranking"] = consensus_strs
    out["matching_rankers"] = matching_rankers_strs
    out["keep_annotation_from"] = keep_from_list
    out["decision_reason"] = reasons

    ensure_dir_for_file(output_csv)
    out.to_csv(output_csv, index=False)

    pass_files = out.loc[out["status"] == "PASS", "filename"].tolist()
    review_files = out.loc[out["status"] == "REVIEW", "filename"].tolist()

    write_list(pass_list_path, pass_files)
    write_list(review_list_path, review_files)

    keep_map = out.loc[
        out["status"] == "PASS",
        ["filename", "keep_annotation_from", "consensus_ranking", "agree_count"],
    ].copy()
    ensure_dir_for_file(keep_map_path)
    keep_map.to_csv(keep_map_path, index=False)

    print(f"Input: {args.input}")
    print(f"Rows read: {len(df)}")
    print(f"Wrote: {output_csv}")
    print(f"Wrote PASS list ({len(pass_files)}): {pass_list_path}")
    print(f"Wrote REVIEW list ({len(review_files)}): {review_list_path}")
    print(f"Wrote keep map ({len(keep_map)}): {keep_map_path}")


if __name__ == "__main__":
    main()
