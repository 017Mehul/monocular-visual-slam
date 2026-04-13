"""Run KITTI ATE benchmark across multiple sequences.

Usage:
  python run_kitti_bench.py --gt-root dataset/dataset/poses --est-root outputs/runs

The script looks for matching files by sequence id (00.txt, 01.txt, ...) in
the ground-truth folder and expected trajectory CSVs named
`trajectory_<seq>.csv` under the estimated root. It prints a CSV summary.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import csv
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "New folder" / "slam"))
import kitti_evaluation as ke


def find_sequences(gt_root: Path):
    return sorted([p.stem for p in gt_root.glob("*.txt")])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt-root", required=True, help="Folder with KITTI poses (*.txt)")
    p.add_argument("--est-root", required=True, help="Folder with estimated trajectories")
    p.add_argument("--out", default="kitti_bench_summary.csv", help="CSV output path")
    args = p.parse_args()

    gt_root = Path(args.gt_root)
    est_root = Path(args.est_root)
    out_path = Path(args.out)

    seqs = find_sequences(gt_root)
    rows = []
    for seq in seqs:
        gt_file = gt_root / f"{seq}.txt"
        est_file = est_root / f"trajectory_{seq}.csv"
        if not est_file.exists():
            print(f"Skipping {seq}: no estimated trajectory at {est_file}")
            continue
        gt_poses = ke.read_kitti_poses(gt_file)
        est_positions = ke.read_positions(est_file)
        ate = ke.compute_ate(gt_poses[:, :3, 3], est_positions)
        rows.append((seq, f"{ate:.6f}"))

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sequence", "ATE_m"])
        writer.writerows(rows)

    print(f"Wrote summary to {out_path}")


if __name__ == "__main__":
    main()
