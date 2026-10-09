"""Runtime health and real-world validation metrics for monocular SLAM."""

from __future__ import annotations

import json
import statistics
import time
from collections import deque
from pathlib import Path


class RuntimeMonitor:
    """Track real-camera performance, tracking health and failure recovery."""

    def __init__(self, window_size: int = 60, max_frame_time_ms: float = 150.0):
        self.window_size = max(1, int(window_size))
        self.max_frame_time_ms = float(max_frame_time_ms)
        self.frame_times_ms = deque(maxlen=self.window_size)
        self.features = deque(maxlen=self.window_size)
        self.matches = deque(maxlen=self.window_size)
        self.inliers = deque(maxlen=self.window_size)
        self.tracked_frames = 0
        self.failed_frames = 0
        self.slow_frames = 0
        self.total_frames = 0
        self.start_time = time.monotonic()

    def record(self, elapsed_sec: float, features: int, matches: int, inliers: int, tracking_ok: bool):
        ms = float(elapsed_sec) * 1000.0
        self.total_frames += 1
        self.tracked_frames += int(tracking_ok)
        self.failed_frames += int(not tracking_ok)
        self.slow_frames += int(ms > self.max_frame_time_ms)
        self.frame_times_ms.append(ms)
        self.features.append(int(features))
        self.matches.append(int(matches))
        self.inliers.append(int(inliers))

    @property
    def elapsed_sec(self):
        return max(time.monotonic() - self.start_time, 1e-9)

    @property
    def effective_fps(self):
        return self.total_frames / self.elapsed_sec

    @property
    def tracking_rate(self):
        return self.tracked_frames / max(self.total_frames, 1)

    @staticmethod
    def _percentile(values, percentile):
        if not values:
            return 0.0
        if len(values) == 1:
            return float(values[0])
        return float(statistics.quantiles(values, n=100, method="inclusive")[percentile - 1])

    def snapshot(self):
        frame_times = list(self.frame_times_ms)
        avg_ms = sum(frame_times) / max(len(frame_times), 1)
        return {
            "frames_observed": self.total_frames,
            "tracking_success_rate": round(self.tracking_rate, 4),
            "tracking_failure_rate": round(self.failed_frames / max(self.total_frames, 1), 4),
            "effective_fps": round(self.effective_fps, 3),
            "avg_frame_time_ms": round(avg_ms, 3),
            "p50_frame_time_ms": round(self._percentile(frame_times, 50), 3),
            "p95_frame_time_ms": round(self._percentile(frame_times, 95), 3),
            "p99_frame_time_ms": round(self._percentile(frame_times, 99), 3),
            "slow_frame_rate": round(self.slow_frames / max(self.total_frames, 1), 4),
            "avg_features": round(sum(self.features) / max(len(self.features), 1), 2),
            "avg_matches": round(sum(self.matches) / max(len(self.matches), 1), 2),
            "avg_inliers": round(sum(self.inliers) / max(len(self.inliers), 1), 2),
            "window_size": self.window_size,
            "max_frame_time_ms": self.max_frame_time_ms,
        }

    def write_json(self, path: str | Path, extra: dict | None = None):
        report = self.snapshot()
        if extra:
            report.update(extra)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
