# map_manager.py - Sparse 3D map of triangulated world points

import numpy as np
from slam.config import PIPELINE_PARAMS


class MapManager:
    """
    Maintains the sparse 3D map.
    """

    def __init__(self):
        self._points: list[np.ndarray] = []   # each entry is shape (3,)
        self.max_points = PIPELINE_PARAMS["max_map_points"]
        self.total_added = 0

    def add_points(self, new_points: np.ndarray):
        if new_points is None or len(new_points) == 0:
            return
        for pt in new_points:
            if np.isfinite(pt).all():
                self._points.append(pt.copy())
                self.total_added += 1
        if len(self._points) > self.max_points:
            self._points = self._points[-self.max_points:]

    def get_point_cloud(self) -> np.ndarray:
        if not self._points:
            return np.empty((0, 3))
        return np.array(self._points)

    def size(self) -> int:
        return len(self._points)

    def prune_outliers(self):
        if len(self._points) < 10:
            return
        cloud = self.get_point_cloud()
        centroid = cloud.mean(axis=0)
        dists = np.linalg.norm(cloud - centroid, axis=1)
        q1, q3 = np.percentile(dists, [25, 75])
        iqr = q3 - q1
        upper = q3 + 3.0 * iqr
        before = len(self._points)
        self._points = [p for p, d in zip(self._points, dists) if d <= upper]
        removed = before - len(self._points)
        if removed > 0:
            print(f"[MapManager] Pruned {removed} outlier points.")
