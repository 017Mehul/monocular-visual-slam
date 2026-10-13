"""Validated per-frame external metric motion for monocular scale recovery.

CSV contract: one row per *current* frame, with the metric displacement in
meters from the immediately preceding frame to this frame. The first frame
usually has no displacement row. This is deliberately not cumulative distance.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def load_metric_scale_csv(path):
    """Return {frame_idx: metric_delta_m}; reject malformed or discontinuous input."""
    samples = {}
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        expected = {"frame_idx", "metric_delta_m"}
        if reader.fieldnames is None or not expected.issubset(set(reader.fieldnames)):
            raise ValueError("CSV header must include frame_idx,metric_delta_m")
        previous_frame = None
        for line_no, row in enumerate(reader, start=2):
            try:
                frame_value = float(row["frame_idx"])
                metric = float(row["metric_delta_m"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric value on CSV line {line_no}") from exc
            if not np.isfinite(frame_value) or not frame_value.is_integer():
                raise ValueError(f"frame_idx must be a finite integer on line {line_no}")
            frame_idx = int(frame_value)
            if frame_idx in samples:
                raise ValueError(f"Duplicate frame_idx {frame_idx} on line {line_no}")
            if not np.isfinite(metric) or metric < 0:
                raise ValueError(f"metric_delta_m must be finite and non-negative on line {line_no}")
            if previous_frame is not None and frame_idx != previous_frame + 1:
                raise ValueError(
                    f"Non-contiguous frame IDs on line {line_no}: expected "
                    f"{previous_frame + 1}, got {frame_idx}"
                )
            samples[frame_idx] = metric
            previous_frame = frame_idx
    if not samples:
        raise ValueError("Metric scale CSV contains no motion samples")
    return samples


class SensorScaleProvider:
    """Estimate metric-units-per-visual-unit using synchronized frame motion."""

    def __init__(self, window=20, min_visual_motion=1e-4):
        if int(window) < 1:
            raise ValueError("window must be >= 1")
        self.window = int(window)
        self.min_visual_motion = float(min_visual_motion)
        self._ratios = []

    def update(self, visual_translation_norm, metric_delta_m):
        visual = float(visual_translation_norm)
        metric = float(metric_delta_m)
        if (not np.isfinite(visual) or not np.isfinite(metric)
                or visual < self.min_visual_motion or metric <= 0):
            return None
        ratio = metric / visual
        if not np.isfinite(ratio) or ratio <= 0:
            return None
        self._ratios.append(float(ratio))
        self._ratios = self._ratios[-self.window:]
        return self.scale

    @property
    def scale(self):
        return float(np.median(self._ratios)) if self._ratios else None

    def reset(self):
        self._ratios.clear()
