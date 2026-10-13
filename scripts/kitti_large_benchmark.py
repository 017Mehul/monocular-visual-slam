#!/usr/bin/env python3
"""Run a frame-aligned, threshold-aware multi-sequence KITTI benchmark."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from slam.kitti_evaluation import compute_ate, read_kitti_poses


def read_estimated_pose_rows(path):
    """Read full-pose CSV as {source_frame_id: 4x4 camera-to-world pose}."""
    rows = {}
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header or len(header) != 13 or header[0].strip() != "frame_idx":
            raise ValueError("Expected trajectory_poses.csv with frame_idx and 12 pose values")
        for line_no, row in enumerate(reader, start=2):
            if len(row) != 13:
                raise ValueError(f"Expected 13 columns on line {line_no}")
            frame_value = float(row[0])
            values = np.asarray([float(v) for v in row[1:]], dtype=float)
            if not np.isfinite(frame_value) or not frame_value.is_integer() or not np.isfinite(values).all():
                raise ValueError(f"Invalid pose or frame ID on line {line_no}")
            frame_id = int(frame_value)
            if frame_id in rows:
                raise ValueError(f"Duplicate estimated frame ID {frame_id}")
            pose = np.eye(4)
            pose[:3, :4] = values.reshape(3, 4)
            rows[frame_id] = pose
    return rows


def aligned_rpe(gt_poses, est_poses, frame_ids, delta=1):
    trans_sq, rot_sq = [], []
    for i in range(len(frame_ids)):
        target = frame_ids[i] + delta
        j = i + 1
        if j >= len(frame_ids) or frame_ids[j] != target:
            continue
        gt_rel = np.linalg.inv(gt_poses[i]) @ gt_poses[j]
        est_rel = np.linalg.inv(est_poses[i]) @ est_poses[j]
        err = np.linalg.inv(gt_rel) @ est_rel
        trans_sq.append(float(np.sum(err[:3, 3] ** 2)))
        angle = np.arccos(np.clip((np.trace(err[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
        rot_sq.append(float(np.degrees(angle) ** 2))
    if not trans_sq:
        return None, None
    return float(np.sqrt(np.mean(trans_sq))), float(np.sqrt(np.mean(rot_sq)))


def run_sequence(sequence_dir, gt_file, output_dir, frames, thresholds):
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_file = sequence_dir / "calib.txt"
    if not calibration_file.is_file():
        return {"sequence": sequence_dir.name, "completed": False, "accuracy_passed": None,
                "status": "FAILED", "error": "sequence-specific calib.txt is missing"}
    cmd = [sys.executable, "slam/main.py", "--source", str(sequence_dir),
           "--headless", "--no-viz", "--save-trajectory", "--summary-json",
           "--output-dir", str(output_dir)]
    if frames is not None:
        cmd += ["--max-frames", str(frames)]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    result = {"sequence": sequence_dir.name, "return_code": proc.returncode}
    if proc.returncode != 0:
        result.update({"completed": False, "accuracy_passed": None, "status": "FAILED",
                       "stderr_tail": proc.stderr[-2000:]})
        return result
    est_file = output_dir / "trajectory_poses.csv"
    if not est_file.exists():
        result.update({"completed": False, "accuracy_passed": None, "status": "FAILED",
                       "error": "trajectory_poses.csv was not produced"})
        return result
    try:
        gt = read_kitti_poses(gt_file)
        est_rows = read_estimated_pose_rows(est_file)
    except (ValueError, OSError) as exc:
        result.update({"completed": False, "accuracy_passed": None, "status": "FAILED",
                       "error": str(exc)})
        return result

    gt_rows = {idx: pose for idx, pose in enumerate(gt)}
    common_ids = sorted(set(gt_rows).intersection(est_rows))
    if len(common_ids) < 2:
        result.update({"completed": False, "accuracy_passed": None, "status": "FAILED",
                       "error": "fewer than two frame-ID-aligned pose pairs"})
        return result
    gt_aligned = np.asarray([gt_rows[idx] for idx in common_ids])
    est_aligned = np.asarray([est_rows[idx] for idx in common_ids])
    ate = compute_ate(gt_aligned[:, :3, 3], est_aligned[:, :3, 3])
    rpe_t, rpe_r = aligned_rpe(gt_aligned, est_aligned, common_ids, delta=1)
    if rpe_t is None:
        result.update({"completed": False, "accuracy_passed": None, "status": "FAILED",
                       "error": "no adjacent frame-ID pairs available for RPE"})
        return result

    checks = {}
    if thresholds["max_ate_rmse_m"] is not None:
        checks["ate_rmse_m"] = ate <= thresholds["max_ate_rmse_m"]
    if thresholds["max_rpe_translation_rmse_m"] is not None:
        checks["rpe_translation_rmse_m"] = rpe_t <= thresholds["max_rpe_translation_rmse_m"]
    if thresholds["max_rpe_rotation_rmse_deg"] is not None:
        checks["rpe_rotation_rmse_deg"] = rpe_r <= thresholds["max_rpe_rotation_rmse_deg"]
    accuracy_passed = all(checks.values()) if checks else None
    result.update({
        "completed": True,
        "accuracy_passed": accuracy_passed,
        "status": "COMPLETED" if accuracy_passed is not False else "ACCURACY_FAILED",
        "samples": len(common_ids),
        "first_aligned_frame_id": common_ids[0],
        "last_aligned_frame_id": common_ids[-1],
        "frame_ids_contiguous_fraction": sum(
            common_ids[i] == common_ids[i - 1] + 1 for i in range(1, len(common_ids))
        ) / max(len(common_ids) - 1, 1),
        "ate_rmse_m": ate,
        "rpe_translation_rmse_m": rpe_t,
        "rpe_rotation_rmse_deg": rpe_r,
        "accuracy_checks": checks,
        "thresholds": thresholds,
    })
    return result


def main():
    p = argparse.ArgumentParser(description="Frame-aligned KITTI multi-sequence benchmark")
    p.add_argument("--dataset-root", required=True, help="KITTI root containing sequences/ and poses/")
    p.add_argument("--sequences", nargs="+", default=None)
    p.add_argument("--frames", type=int, default=None)
    p.add_argument("--output-dir", default="outputs/kitti_benchmark")
    p.add_argument("--max-ate-rmse-m", type=float, default=None)
    p.add_argument("--max-rpe-translation-rmse-m", type=float, default=None)
    p.add_argument("--max-rpe-rotation-rmse-deg", type=float, default=None)
    p.add_argument("--fail-fast", action="store_true")
    args = p.parse_args()
    for name in ("max_ate_rmse_m", "max_rpe_translation_rmse_m", "max_rpe_rotation_rmse_deg"):
        value = getattr(args, name)
        if value is not None and value < 0:
            p.error(f"--{name.replace('_', '-')} must be non-negative")
    root = Path(args.dataset_root)
    seq_root, pose_root = root / "sequences", root / "poses"
    if not seq_root.is_dir() or not pose_root.is_dir():
        p.error("dataset-root must contain sequences/ and poses/ directories")
    sequences = args.sequences or sorted(x.name for x in seq_root.iterdir() if x.is_dir())
    thresholds = {
        "max_ate_rmse_m": args.max_ate_rmse_m,
        "max_rpe_translation_rmse_m": args.max_rpe_translation_rmse_m,
        "max_rpe_rotation_rmse_deg": args.max_rpe_rotation_rmse_deg,
    }
    results = []
    for seq in sequences:
        seq_dir, gt_file = seq_root / seq, pose_root / f"{seq}.txt"
        if not seq_dir.is_dir() or not gt_file.is_file():
            result = {"sequence": seq, "completed": False, "accuracy_passed": None,
                      "status": "SKIPPED", "error": "missing sequence or ground truth"}
        else:
            print(f"[benchmark] running KITTI {seq}")
            result = run_sequence(seq_dir, gt_file, Path(args.output_dir) / seq, args.frames, thresholds)
        results.append(result)
        print(json.dumps(result))
        if args.fail_fast and result["status"] in ("FAILED", "ACCURACY_FAILED", "SKIPPED"):
            break
    completed = [r for r in results if r.get("completed")]
    accuracy_evaluated = [r for r in completed if r.get("accuracy_passed") is not None]
    summary = {
        "dataset_root": str(root),
        "sequences_requested": sequences,
        "sequences_completed": len(completed),
        "sequences_accuracy_passed": sum(r["accuracy_passed"] is True for r in accuracy_evaluated),
        "accuracy_thresholds_configured": any(v is not None for v in thresholds.values()),
        "mean_ate_rmse_m": float(np.mean([r["ate_rmse_m"] for r in completed])) if completed else None,
        "mean_rpe_translation_rmse_m": float(np.mean([r["rpe_translation_rmse_m"] for r in completed])) if completed else None,
        "mean_rpe_rotation_rmse_deg": float(np.mean([r["rpe_rotation_rmse_deg"] for r in completed])) if completed else None,
        "thresholds": thresholds,
        "results": results,
    }
    out = Path(args.output_dir) / "benchmark_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    has_failures = any(r["status"] in ("FAILED", "ACCURACY_FAILED", "SKIPPED") for r in results)
    return 0 if completed and not has_failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
