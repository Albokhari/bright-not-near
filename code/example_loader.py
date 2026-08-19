"""Minimal example: join the released annotations with Kvasir-SEG images.

Usage:
    python example_loader.py /path/to/Kvasir-SEG/images

Prints per-image points/ranks and iterates the brightness-discordant pairs.
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ANN = os.path.join(HERE, "..", "annotations")


def load_benchmark():
    """307 images -> list of {filename, width, height, points:[{point_id,x,y,rank}]}"""
    with open(os.path.join(ANN, "consensus_307_points_ranks.json")) as f:
        return json.load(f)["images"]


def load_discordant_pairs():
    """391 brightness-discordant pairs -> list of (filename, point_id_a, point_id_b)"""
    with open(os.path.join(ANN, "discordant_pairs.csv")) as f:
        rows = list(csv.DictReader(f))
    return [(r["filename"], int(r["point_id_a"]), int(r["point_id_b"])) for r in rows]


def main(image_root):
    images = load_benchmark()
    pairs = load_discordant_pairs()
    print(f"{len(images)} benchmark images, {len(pairs)} discordant pairs")

    im = images[0]
    path = os.path.join(image_root, im["filename"])
    print(f"\nexample image: {path} (exists: {os.path.exists(path)})")
    for p in sorted(im["points"], key=lambda p: p["rank"]):
        print(f"  rank {p['rank']}  point {p['point_id']}  at ({p['x']:.0f}, {p['y']:.0f})")

    # score a model on the discordant pairs: for each (filename, a, b), your model's
    # predicted nearness at point a should exceed that at point b iff rank_a < rank_b.
    by_name = {im["filename"]: im for im in images}
    fn, a, b = pairs[0]
    pts = {p["point_id"]: p for p in by_name[fn]["points"]}
    print(f"\nfirst discordant pair: {fn} points {a},{b} "
          f"(ranks {pts[a]['rank']},{pts[b]['rank']})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
