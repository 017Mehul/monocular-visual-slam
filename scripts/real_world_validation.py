#!/usr/bin/env python3
"""Run a repeatable real-camera/headless validation pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slam.main import run


def main():
    parser = argparse.ArgumentParser(description="Real-world monocular SLAM validation")
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--calibration", default=None, help="Calibration .npz produced by slam/calibration.py")
    parser.add_argument("--frames", type=int, default=300, help="Frames to validate")
    parser.add_argument("--output-dir", default="outputs/real_world_validation")
    parser.add_argument("--min-tracking-rate", type=float, default=0.70)
    parser.add_argument("--min-fps", type=float, default=5.0)
    parser.add_argument("--max-slow-frame-rate", type=float, default=0.50)
    parser.add_argument("--max-p95-frame-time-ms", type=float, default=150.0)
    args = parser.parse_args()

    from argparse import Namespace
    run_args = Namespace(
        source=args.source,
        scale=0.5,
        width=None,
        height=None,
        no_viz=True,
        headless=True,
        no_ba=True,
        max_frames=args.frames,
        config_file="config.production.json",
        calibration=args.calibration,
        output_dir=args.output_dir,
        save_trajectory=True,
        summary_json=True,
        log_level="INFO",
        log_file=None,
        metrics_file=str(Path(args.output_dir) / "runtime_metrics.json"),
    )
    summary = run(run_args)
    metrics = summary["runtime_validation"]
    failures = []
    if metrics["p95_frame_time_ms"] > args.max_p95_frame_time_ms:
        failures.append("p95_frame_time_ms=%.2f > %.2f" % (metrics["p95_frame_time_ms"], args.max_p95_frame_time_ms))
    if metrics["tracking_success_rate"] < args.min_tracking_rate:
        failures.append(f"tracking_success_rate={metrics['tracking_success_rate']:.3f} < {args.min_tracking_rate:.3f}")
    if metrics["effective_fps"] < args.min_fps:
        failures.append(f"effective_fps={metrics['effective_fps']:.2f} < {args.min_fps:.2f}")
    if metrics["slow_frame_rate"] > args.max_slow_frame_rate:
        failures.append(f"slow_frame_rate={metrics['slow_frame_rate']:.3f} > {args.max_slow_frame_rate:.3f}")

    report = {
        "status": "PASS" if not failures else "FAIL",
        "thresholds": {
            "min_tracking_rate": args.min_tracking_rate,
            "min_fps": args.min_fps,
            "max_slow_frame_rate": args.max_slow_frame_rate,
            "max_p95_frame_time_ms": args.max_p95_frame_time_ms,
        },
        "metrics": metrics,
        "failures": failures,
    }
    path = Path(args.output_dir) / "validation_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
