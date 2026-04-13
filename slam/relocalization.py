# relocalization.py - Lost tracking recovery

import cv2
import numpy as np
from enum import Enum
from slam.pose_estimation import build_intrinsic_matrix


class TrackingState(Enum):
    TRACKING = "TRACKING"
    LOST = "LOST"


class RelocFrame:
    def __init__(self, frame_idx, keypoints, descriptors, points_3d, R, t):
        self.frame_idx = frame_idx
        self.keypoints = keypoints
        self.descriptors = descriptors.astype(np.uint8)
        self.points_3d = points_3d
        self.R = R
        self.t = t
        self._mean = descriptors.astype(np.float32).mean(axis=0)


class Relocalizer:
    def __init__(
        self,
        lost_threshold: int = 5,
        min_pnp_inliers: int = 12,
        sim_threshold: float = 0.70,
    ):
        self.lost_threshold = lost_threshold
        self.min_pnp_inliers = min_pnp_inliers
        self.sim_threshold = sim_threshold

        self.state = TrackingState.TRACKING
        self._fail_count = 0
        self._db: list[RelocFrame] = []

        self.K = build_intrinsic_matrix()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def update_state(self, pose_ok: bool) -> TrackingState:
        if pose_ok:
            self._fail_count = 0
            self.state = TrackingState.TRACKING
        else:
            self._fail_count += 1
            if self._fail_count >= self.lost_threshold:
                if self.state != TrackingState.LOST:
                    print("[Relocalizer] Tracking LOST — attempting relocalization.")
                self.state = TrackingState.LOST
        return self.state

    def is_lost(self) -> bool:
        return self.state == TrackingState.LOST

    def add_frame(self, frame_idx, keypoints, descriptors, points_3d, R, t):
        if descriptors is None or points_3d is None or len(points_3d) < 4:
            return
        self._db.append(RelocFrame(frame_idx, keypoints, descriptors, points_3d, R, t))

    def relocalize(self, keypoints, descriptors):
        if descriptors is None or len(self._db) == 0:
            return None, None, False
        curr_mean = descriptors.astype(np.float32).mean(axis=0)
        candidates = self._get_candidates(curr_mean, top_k=5)
        for cand in candidates:
            R, t, ok = self._pnp_verify(keypoints, descriptors, cand)
            if ok:
                self.state = TrackingState.TRACKING
                self._fail_count = 0
                print(f"[Relocalizer] Relocalized against frame {cand.frame_idx}.")
                return R, t, True
        return None, None, False

    def _get_candidates(self, curr_mean, top_k=5):
        sims = []
        for rf in self._db:
            d = np.linalg.norm(curr_mean) * np.linalg.norm(rf._mean)
            sim = float(np.dot(curr_mean, rf._mean) / d) if d > 1e-8 else 0.0
            sims.append((sim, rf))
        sims.sort(key=lambda x: -x[0])
        return [rf for sim, rf in sims[:top_k] if sim >= self.sim_threshold]

    def _pnp_verify(self, kp_curr, desc_curr, cand: RelocFrame):
        raw = self.matcher.knnMatch(desc_curr, cand.descriptors, k=2)
        good = [m for m, n in raw if len([m, n]) == 2 and m.distance < 0.75 * n.distance]
        if len(good) < self.min_pnp_inliers:
            return None, None, False
        pts_2d, pts_3d = [], []
        for m in good:
            if m.trainIdx < len(cand.points_3d):
                pts_2d.append(kp_curr[m.queryIdx].pt)
                pts_3d.append(cand.points_3d[m.trainIdx])
        if len(pts_3d) < self.min_pnp_inliers:
            return None, None, False
        pts_2d = np.array(pts_2d, dtype=np.float64)
        pts_3d = np.array(pts_3d, dtype=np.float64)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, self.K, None,
            iterationsCount=200,
            reprojectionError=4.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok or inliers is None or len(inliers) < self.min_pnp_inliers:
            return None, None, False
        R, _ = cv2.Rodrigues(rvec)
        return R, tvec.reshape(3, 1), True
