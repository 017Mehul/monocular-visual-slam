"""External metric-scale adapters for monocular SLAM."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class MetricScaleSample:
    frame_idx: int
    distance_m: float

class SensorScaleProvider:
    """Fuse metric motion from wheel/IMU/GNSS with visual translation."""
    def __init__(self, window=20, min_visual_motion=1e-4):
        self.window = int(window)
        self.min_visual_motion = float(min_visual_motion)
        self._ratios = []

    def update(self, visual_translation_norm, metric_translation_m):
        visual = float(visual_translation_norm)
        metric = float(metric_translation_m)
        if not np.isfinite(visual) or not np.isfinite(metric) or visual < self.min_visual_motion or metric <= 0:
            return self.scale
        ratio = metric / visual
        if not np.isfinite(ratio) or ratio <= 0:
            return self.scale
        self._ratios.append(float(ratio))
        self._ratios = self._ratios[-self.window:]
        return self.scale

    @property
    def scale(self):
        return float(np.median(self._ratios)) if self._ratios else None

    def reset(self):
        self._ratios.clear()

def load_metric_scale_csv(path):
    """Load rows of frame_idx,metric_translation_m."""
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    if data.shape[1] < 2:
        raise ValueError("Metric scale CSV must contain frame_idx,metric_translation_m")
    return {int(row[0]): float(row[1]) for row in data if np.isfinite(row[0]) and np.isfinite(row[1])}
