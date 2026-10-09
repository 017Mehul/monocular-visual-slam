# triangulation.py - Triangulate matched 2D points into world-frame 3D map points

import cv2
import numpy as np
from slam.pose_estimation import build_intrinsic_matrix
from slam.config import TRIANGULATION_PARAMS


class Triangulator:
    """Triangulate corresponding image points using world-to-camera poses.

    R/t passed to this class are world-to-camera extrinsics:
        X_cam = R @ X_world + t
    The returned points are in the world coordinate frame.
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
        return_mask: bool = False,
    ):
        pts1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 2)
        pts2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 2)
        n = min(len(pts1), len(pts2))
        if n < 4:
            empty = np.empty((0, 3), dtype=np.float64)
            return (empty, np.zeros(n, dtype=bool)) if return_mask else empty
        pts1, pts2 = pts1[:n], pts2[:n]

        R1 = np.asarray(R1, dtype=np.float64).reshape(3, 3)
        t1 = np.asarray(t1, dtype=np.float64).reshape(3, 1)
        R2 = np.asarray(R2, dtype=np.float64).reshape(3, 3)
        t2 = np.asarray(t2, dtype=np.float64).reshape(3, 1)

        P1 = self.K @ np.hstack([R1, t1])
        P2 = self.K @ np.hstack([R2, t2])
        pts_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

        w = pts_4d[3]
        finite_w = np.isfinite(w) & (np.abs(w) > 1e-8)
        points = np.full((n, 3), np.nan, dtype=np.float64)
        points[finite_w] = (pts_4d[:3, finite_w] / w[finite_w]).T

        valid = finite_w & np.isfinite(points).all(axis=1)
        if valid.any():
            idx = np.flatnonzero(valid)
            p = points[idx]
            depth1 = (R1 @ p.T + t1).T[:, 2]
            depth2 = (R2 @ p.T + t2).T[:, 2]
            ratio = np.linalg.norm(p[:, :2], axis=1) / (np.abs(depth1) + 1e-8)
            good = (
                (depth1 > self.min_depth)
                & (depth1 < self.max_depth)
                & (depth2 > self.min_depth)
                & (depth2 < self.max_depth)
                & (ratio < 10.0)
            )
            valid[idx[~good]] = False

        result = points[valid]
        if return_mask:
            return result, valid
        return result
