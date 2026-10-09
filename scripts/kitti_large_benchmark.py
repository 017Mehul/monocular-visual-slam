#!/usr/bin/env python3
"""Run and aggregate a reproducible multi-sequence KITTI benchmark."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
from slam.kitti_evaluation import compute_ate, compute_rpe, read_estimated_poses, read_kitti_poses

def run_sequence(sequence_dir, gt_file, output_dir, frames):
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "slam/main.py", "--source", str(sequence_dir),
           "--headless", "--no-viz", "--save-trajectory", "--summary-json",
           "--output-dir", str(output_dir)]
    if frames:
        cmd += ["--max-frames", str(frames)]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    result = {"sequence": sequence_dir.name, "return_code": proc.returncode}
    if proc.returncode != 0:
        result.update({"status": "FAILED", "stderr_tail": proc.stderr[-2000:]})
        return result
    est_file = output_dir / "trajectory_poses.csv"
    if not est_file.exists():
        result.update({"status": "FAILED", "error": "trajectory_poses.csv was not produced"})
        return result
    gt = read_kitti_poses(gt_file)
    est = read_estimated_poses(est_file)
    n = min(len(gt), len(est))
    if n < 2:
        result.update({"status": "FAILED", "error": "insufficient trajectory samples"})
        return result
    ate = compute_ate(gt[:n, :3, 3], est[:n, :3, 3])
    rpe_t, rpe_r = compute_rpe(gt[:n], est[:n], delta=1)
    result.update({"status": "PASS", "samples": n, "ate_rmse_m": ate,
                   "rpe_translation_rmse_m": rpe_t, "rpe_rotation_rmse_deg": rpe_r})
    return result

def main():
    p = argparse.ArgumentParser(description="Large-scale KITTI multi-sequence benchmark")
    p.add_argument("--dataset-root", required=True, help="KITTI root containing sequences/ and poses/")
    p.add_argument("--sequences", nargs="+", default=None)
    p.add_argument("--frames", type=int, default=None)
    p.add_argument("--output-dir", default="outputs/kitti_benchmark")
    p.add_argument("--fail-fast", action="store_true")
    args = p.parse_args()
    root = Path(args.dataset_root)
    seq_root, pose_root = root / "sequences", root / "poses"
    sequences = args.sequences or sorted(p.name for p in seq_root.iterdir() if p.is_dir())
    results = []
    for seq in sequences:
        seq_dir, gt_file = seq_root / seq, pose_root / f"{seq}.txt"
        if not seq_dir.exists() or not gt_file.exists():
            results.append({"sequence": seq, "status": "SKIPPED", "error": "missing sequence or ground truth"})
            if args.fail_fast:
                break
            continue
        print(f"[benchmark] running KITTI {seq}")
        result = run_sequence(seq_dir, gt_file, Path(args.output_dir) / seq, args.frames)
        results.append(result)
        print(json.dumps(result))
        if result["status"] == "FAILED" and args.fail_fast:
            break
    passed = [r for r in results if r["status"] == "PASS"]
    summary = {
        "dataset_root": str(root), "sequences_requested": sequences,
        "sequences_passed": len(passed),
        "sequences_evaluated": len([r for r in results if r["status"] != "SKIPPED"]),
        "mean_ate_rmse_m": sum(r["ate_rmse_m"] for r in passed) / len(passed) if passed else None,
        "mean_rpe_translation_rmse_m": sum(r["rpe_translation_rmse_m"] for r in passed) / len(passed) if passed else None,
        "mean_rpe_rotation_rmse_deg": sum(r["rpe_rotation_rmse_deg"] for r in passed) / len(passed) if passed else None,
        "results": results,
    }
    out = Path(args.output_dir) / "benchmark_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    failed = [r for r in results if r["status"] == "FAILED"]
    return 0 if passed and not failed else 2

if __name__ == "__main__":
    raise SystemExit(main())
