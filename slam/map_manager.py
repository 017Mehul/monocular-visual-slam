# map_manager.py - Sparse 3D map of world-frame landmarks

import numpy as np
from slam.config import PIPELINE_PARAMS


class MapManager:
    def __init__(self):
        self._points = []
        self.max_points = PIPELINE_PARAMS["max_map_points"]
        self.total_added = 0

    def add_points(self, new_points):
        if new_points is None or len(new_points) == 0:
            return
        for pt in np.asarray(new_points):
            if np.isfinite(pt).all():
                self._points.append(np.asarray(pt, dtype=np.float64).copy())
                self.total_added += 1
        if len(self._points) > self.max_points:
            self._points = self._points[-self.max_points:]

    def replace_points(self, points):
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        self._points = [p.copy() for p in pts if np.isfinite(p).all()][-self.max_points:]

    def get_point_cloud(self):
        return np.asarray(self._points, dtype=np.float64).reshape(-1, 3) if self._points else np.empty((0, 3))

    def size(self):
        return len(self._points)

    def prune_outliers(self):
        if len(self._points) < 10:
            return
        cloud = self.get_point_cloud()
        centroid = cloud.mean(axis=0)
        dists = np.linalg.norm(cloud - centroid, axis=1)
        q1, q3 = np.percentile(dists, [25, 75])
        upper = q3 + 3.0 * (q3 - q1)
        self._points = [p for p, d in zip(self._points, dists) if d <= upper]
