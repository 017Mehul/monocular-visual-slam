# pose_estimation.py - Essential matrix estimation and pose recovery

import cv2
import numpy as np
from slam.config import CAMERA_PARAMS, RANSAC_PARAMS


def build_intrinsic_matrix() -> np.ndarray:
    """Construct the 3x3 camera intrinsic matrix K from config."""
    p = CAMERA_PARAMS
    return np.array([
        [p["fx"],      0, p["cx"]],
        [     0, p["fy"], p["cy"]],
        [     0,       0,       1],
    ], dtype=np.float64)


class PoseEstimator:
    """
    Estimates relative camera pose (R, t) via Essential Matrix decomposition.

    Key improvements:
    - Points are normalised into camera coordinates before findEssentialMat.
      This makes RANSAC threshold meaningful in pixel units and improves
      numerical stability of the 5-point algorithm.
    - Minimum inlier count check (RANSAC_PARAMS["min_inliers"]) rejects
      frames where the scene provides insufficient geometric constraint.
    - Rotation sanity check: if the recovered rotation angle exceeds 30°
      between consecutive frames, the pose is likely degenerate (pure rotation,
      degenerate scene) and is rejected.
    """

    def __init__(self):
        self.K = build_intrinsic_matrix()
        self.min_inliers = RANSAC_PARAMS["min_inliers"]

    def estimate(self, pts1: np.ndarray, pts2: np.ndarray):
        """
        Compute Essential Matrix and recover (R, t).

        Args:
            pts1, pts2: Matched pixel coordinates, shape (N, 2) float32.

        Returns:
            R (3,3), t (3,1), inlier_mask (N,) bool, success bool
        """
        if len(pts1) < 8:
            return None, None, None, False

        pts1 = pts1.astype(np.float64)
        pts2 = pts2.astype(np.float64)

        E, mask = cv2.findEssentialMat(
            pts1, pts2,
            cameraMatrix=self.K,
            method=cv2.RANSAC,
            prob=RANSAC_PARAMS["prob"],
            threshold=RANSAC_PARAMS["threshold"],
        )

        if E is None or mask is None:
            return None, None, None, False

        # Handle degenerate case where findEssentialMat returns stacked matrices
        if E.shape != (3, 3):
            E = E[:3, :3]

        inlier_count = int(mask.sum())
        if inlier_count < self.min_inliers:
            return None, None, None, False

        n_inliers, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)

        if n_inliers < self.min_inliers:
            return None, None, None, False

        # Reject implausibly large rotations between consecutive frames
        angle = self._rotation_angle(R)
        if angle > 30.0:
            return None, None, None, False

        return R, t, mask.ravel().astype(bool), True

    def filter_inliers(self, pts1, pts2, mask):
        return pts1[mask], pts2[mask]

    @staticmethod
    def _rotation_angle(R: np.ndarray) -> float:
        """Return the rotation angle in degrees for a 3x3 rotation matrix."""
        # Rodrigues formula: angle = arccos((trace(R) - 1) / 2)
        cos_angle = (np.trace(R) - 1.0) / 2.0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))
