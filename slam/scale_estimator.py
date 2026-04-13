# scale_estimator.py - Relative scale recovery for monocular SLAM

import numpy as np


class ScaleEstimator:
    def __init__(self, min_points: int = 8, scale_clip: float = 5.0):
        self.min_points = min_points
        self.scale_clip = scale_clip
        self._last_scale = 1.0

    def estimate(
        self,
        pts3d_prev: np.ndarray,
        pts3d_curr: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
    ) -> float:
        if (pts3d_prev is None or pts3d_curr is None
                or len(pts3d_prev) < self.min_points
                or len(pts3d_curr) < self.min_points):
            return self._last_scale

        depth_prev = self._positive_depths(pts3d_prev)

        R = np.array(R, dtype=np.float64)
        t = np.array(t, dtype=np.float64).ravel()
        pts_transformed = (R @ pts3d_curr.T).T + t
        depth_curr = self._positive_depths(pts_transformed)

        if len(depth_prev) < self.min_points or len(depth_curr) < self.min_points:
            return self._last_scale

        scale = np.median(depth_prev) / (np.median(depth_curr) + 1e-8)
        scale = float(np.clip(scale, 1.0 / self.scale_clip, self.scale_clip))
        scale = 0.7 * self._last_scale + 0.3 * scale
        self._last_scale = scale
        return scale

    @staticmethod
    def _positive_depths(pts: np.ndarray) -> np.ndarray:
        depths = pts[:, 2]
        return depths[depths > 0.01]

    def reset(self):
        self._last_scale = 1.0
