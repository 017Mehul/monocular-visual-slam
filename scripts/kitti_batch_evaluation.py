#!/usr/bin/env python3
"""Batch-evaluate multiple KITTI estimated trajectories with consistent metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slam.kitti_evaluation import (
    compute_ate,
    compute_rpe,
    read_estimated_poses,
    read_kitti_poses,
    read_positions,
)


def evaluate(gt_path: Path, est_path: Path, delta: int):
    gt = read_kitti_poses(gt_path)
    est_poses = read_estimated_poses(est_path)
    est_positions = est_poses[:, :3, 3] if est_poses is not None else read_positions(est_path)
    result = {"estimated_file": str(est_path), "ate_rmse_m": compute_ate(gt[:, :3, 3], est_positions)}
    if est_poses is not None:
        trans, rot = compute_rpe(gt, est_poses, delta=delta)
        result.update({"rpe_translation_rmse_m": trans, "rpe_rotation_rmse_deg": rot})
    return result


def main():
    p = argparse.ArgumentParser(description="Batch KITTI ATE/RPE evaluation")
    p.add_argument("--gt", required=True, help="KITTI ground-truth pose file")
    p.add_argument("--est-dir", required=True, help="Directory containing trajectory CSV files")
    p.add_argument("--pattern", default="**/trajectory_poses.csv")
    p.add_argument("--rpe-delta", type=int, default=1)
    p.add_argument("--output", default="outputs/kitti_batch_metrics.json")
    args = p.parse_args()

    gt = Path(args.gt)
    files = sorted(Path(args.est_dir).glob(args.pattern))
    if not files:
        raise SystemExit(f"No estimated trajectories matched {args.pattern!r} under {args.est_dir}")

    results = [evaluate(gt, path, args.rpe_delta) for path in files]
    summary = {
        "ground_truth": str(gt),
        "count": len(results),
        "results": results,
        "mean_ate_rmse_m": sum(r["ate_rmse_m"] for r in results) / len(results),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
