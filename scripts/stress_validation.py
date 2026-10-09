#!/usr/bin/env python3
"""Run a bounded long-duration SLAM stability test and enforce health gates."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Long-duration SLAM stress validation")
    p.add_argument("--source", required=True, help="Camera index or recorded video path")
    p.add_argument("--calibration", default=None)
    p.add_argument("--frames", type=int, default=3000)
    p.add_argument("--output-dir", default="outputs/stress_validation")
    p.add_argument("--min-tracking-rate", type=float, default=0.70)
    p.add_argument("--min-fps", type=float, default=5.0)
    p.add_argument("--max-slow-frame-rate", type=float, default=0.50)
    p.add_argument("--max-p95-frame-time-ms", type=float, default=150.0)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "scripts/real_world_validation.py",
        "--source", args.source, "--frames", str(args.frames),
        "--output-dir", str(out),
        "--min-tracking-rate", str(args.min_tracking_rate),
        "--min-fps", str(args.min_fps),
        "--max-slow-frame-rate", str(args.max_slow_frame_rate),
        "--max-p95-frame-time-ms", str(args.max_p95_frame_time_ms),
    ]
    if args.calibration:
        cmd += ["--calibration", args.calibration]

    result = subprocess.run(cmd, check=False)
    report_path = out / "validation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {
        "status": "FAIL",
        "failures": ["validation runner did not produce validation_report.json"],
    }
    report["stress_frames_requested"] = args.frames
    report["stress_command_exit_code"] = result.returncode
    report["status"] = "PASS" if result.returncode == 0 and report.get("status") == "PASS" else "FAIL"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
