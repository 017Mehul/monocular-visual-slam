# triangulation.py - Triangulate matched 2D points into 3D map points

import cv2
import numpy as np
from slam.pose_estimation import build_intrinsic_matrix
from slam.config import TRIANGULATION_PARAMS


class Triangulator:
    """
    Triangulates 3D points from two camera views via cv2.triangulatePoints (DLT).
    """

    def __init__(self):
        self.K = build_intrinsic_matrix()
        self.min_depth = TRIANGULATION_PARAMS["min_depth"]
        self.max_depth = TRIANGULATION_PARAMS["max_depth"]

    def triangulate(
        self,
        R1: np.ndarray, t1: np.ndarray,
        R2: np.ndarray, t2: np.ndarray,
        pts1: np.ndarray, pts2: np.ndarray,
    ) -> np.ndarray:
        if len(pts1) < 4 or len(pts2) < 4:
            return np.empty((0, 3))

        R1 = np.array(R1, dtype=np.float64).reshape(3, 3)
        t1 = np.array(t1, dtype=np.float64).reshape(3, 1)
        R2 = np.array(R2, dtype=np.float64).reshape(3, 3)
        t2 = np.array(t2, dtype=np.float64).reshape(3, 1)

        P1 = self.K @ np.hstack([R1, t1])
        P2 = self.K @ np.hstack([R2, t2])

        pts1_T = np.asarray(pts1, dtype=np.float64).reshape(-1, 2).T
        pts2_T = np.asarray(pts2, dtype=np.float64).reshape(-1, 2).T

        pts_4d = cv2.triangulatePoints(P1, P2, pts1_T, pts2_T)

        w = pts_4d[3]
        valid_w = np.abs(w) > 1e-7
        pts_4d = pts_4d[:, valid_w]
        if pts_4d.shape[1] == 0:
            return np.empty((0, 3))

        pts_3d = (pts_4d[:3] / pts_4d[3]).T

        return self._filter_by_depth(pts_3d, R1, t1)

    def _filter_by_depth(self, pts_3d, R, t):
        pts_cam = (R @ pts_3d.T + t).T
        depth = pts_cam[:, 2]
        valid = (depth > self.min_depth) & (depth < self.max_depth)
        valid &= np.isfinite(pts_3d).all(axis=1)
        valid &= pts_cam[:, 2] > 0
        pts_valid = pts_3d[valid]
        if len(pts_valid) == 0:
            return pts_valid
        pts_cam_valid = pts_cam[valid]
        ratio = np.linalg.norm(pts_cam_valid[:, :2], axis=1) / (pts_cam_valid[:, 2] + 1e-8)
        keep = ratio < 10.0
        return pts_valid[keep]
