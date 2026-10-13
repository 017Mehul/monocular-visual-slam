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
        self.descriptors = np.asarray(descriptors, dtype=np.uint8)
        self.points_3d = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
        self.R = R
        self.t = t


class Relocalizer:
    def __init__(
        self,
        lost_threshold: int = 5,
        min_pnp_inliers: int = 12,
        sim_threshold: float = 0.70,
    ):
        self.lost_threshold = lost_threshold
        self.min_pnp_inliers = min_pnp_inliers
        # Retained for backwards-compatible construction; descriptor matching,
        # not a cosine score over averaged binary bytes, ranks candidates.
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
        descriptors = np.asarray(descriptors, dtype=np.uint8)
        points_3d = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
        count = min(len(descriptors), len(points_3d), len(keypoints))
        if count < 4:
            return
        # The descriptor at index i must refer to the same 3D landmark i.
        self._db.append(
            RelocFrame(frame_idx, list(keypoints)[:count], descriptors[:count],
                       points_3d[:count], R, t)
        )

    def sync_keyframes(self, keyframes):
        """Refresh database landmarks after local bundle adjustment."""
        by_frame = {kf.frame_idx: kf for kf in keyframes}
        for record in self._db:
            kf = by_frame.get(record.frame_idx)
            if kf is None:
                continue
            count = min(len(kf.descriptors), len(kf.points_3d), len(kf.keypoints))
            if count < 4:
                continue
            record.keypoints = list(kf.keypoints[:count])
            record.descriptors = np.asarray(kf.descriptors[:count], dtype=np.uint8).copy()
            record.points_3d = np.asarray(kf.points_3d[:count], dtype=np.float64).copy()
            record.R = np.asarray(kf.R, dtype=np.float64).copy()
            record.t = np.asarray(kf.t, dtype=np.float64).reshape(3, 1).copy()

    def apply_world_corrections(self, corrections_by_frame_idx):
        """Keep all stored relocalization landmarks in the corrected world frame."""
        for record in self._db:
            correction = corrections_by_frame_idx.get(record.frame_idx)
            if correction is None:
                continue
            C = np.asarray(correction, dtype=np.float64).reshape(4, 4)
            T_world_to_camera = np.eye(4)
            T_world_to_camera[:3, :3] = np.asarray(record.R, dtype=np.float64)
            T_world_to_camera[:3, 3] = np.asarray(record.t, dtype=np.float64).reshape(3)
            corrected = T_world_to_camera @ np.linalg.inv(C)
            record.R = corrected[:3, :3].copy()
            record.t = corrected[:3, 3].reshape(3, 1).copy()
            if len(record.points_3d):
                homogeneous = np.column_stack([
                    record.points_3d, np.ones(len(record.points_3d))
                ])
                record.points_3d = (C @ homogeneous.T).T[:, :3]

    def estimate_pose_from_map(self, keypoints, descriptors, top_k=3):
        """Estimate an absolute world-to-camera pose from recent mapped landmarks.

        PnP ties the current pose to existing 3D map points; relative VO remains
        the fallback when map support is insufficient.
        """
        if descriptors is None or len(self._db) == 0:
            return None, None, 0
        best_R, best_t, best_count = None, None, 0
        for cand in reversed(self._db[-max(1, int(top_k)):]):
            R, t, count = self._pnp_pose(keypoints, descriptors, cand)
            if count > best_count:
                best_R, best_t, best_count = R, t, count
            # A strong recent-frame solution is already well constrained; avoid
            # extra descriptor matching against older keyframes on every frame.
            if best_count >= max(30, 2 * self.min_pnp_inliers):
                break
        return best_R, best_t, best_count
    def relocalize(self, keypoints, descriptors):
        if descriptors is None or len(self._db) == 0:
            return None, None, False
        candidates = self._get_candidates(descriptors, top_k=10)
        for cand in candidates:
            R, t, ok = self._pnp_verify(keypoints, descriptors, cand)
            if ok:
                self.state = TrackingState.TRACKING
                self._fail_count = 0
                print(f"[Relocalizer] Relocalized against frame {cand.frame_idx}.")
                return R, t, True
        return None, None, False

    def _good_matches(self, descriptors, candidate_descriptors):
        if descriptors is None or candidate_descriptors is None or len(candidate_descriptors) < 2:
            return []
        pairs = self.matcher.knnMatch(
            np.asarray(descriptors, dtype=np.uint8),
            np.asarray(candidate_descriptors, dtype=np.uint8),
            k=2,
        )
        good = []
        for pair in pairs:
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                good.append(pair[0])
        return good

    def _get_candidates(self, descriptors, top_k=10):
        """Rank candidate keyframes by ratio-tested ORB correspondences.

        A cosine similarity between means of packed ORB bytes is not a valid
        visual descriptor metric; it caused valid recovery candidates to be
        filtered out after abrupt motion or a short tracking failure.
        """
        ranked = []
        query = np.asarray(descriptors, dtype=np.uint8)
        # Cheap retrieval uses only the strongest ~256 ORB descriptors per
        # image. Full descriptor sets are reserved for geometric PnP checks
        # on the top candidates, avoiding O(database * all-features^2) work
        # during every frame of a prolonged tracking-loss episode.
        query_step = max(1, len(query) // 256)
        query_sample = query[::query_step][:256]
        for rf in self._db:
            cand = rf.descriptors
            cand_step = max(1, len(cand) // 256)
            candidate_sample = cand[::cand_step][:256]
            good = self._good_matches(query_sample, candidate_sample)
            if len(good) >= min(self.min_pnp_inliers, 12):
                ranked.append((len(good), rf))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [rf for _, rf in ranked[:top_k]]

    def _pnp_pose(self, kp_curr, desc_curr, cand: RelocFrame):
        """Return an absolute world-to-camera pose and RANSAC inlier count."""
        good = self._good_matches(desc_curr, cand.descriptors)
        if len(good) < self.min_pnp_inliers:
            return None, None, 0
        pts_2d, pts_3d = [], []
        for match in good:
            if match.trainIdx < len(cand.points_3d) and match.queryIdx < len(kp_curr):
                pts_2d.append(kp_curr[match.queryIdx].pt)
                pts_3d.append(cand.points_3d[match.trainIdx])
        if len(pts_3d) < self.min_pnp_inliers:
            return None, None, 0
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.asarray(pts_3d, dtype=np.float64),
            np.asarray(pts_2d, dtype=np.float64),
            self.K, None, iterationsCount=200,
            reprojectionError=3.0, confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        count = int(len(inliers)) if ok and inliers is not None else 0
        if count < self.min_pnp_inliers:
            return None, None, 0
        R, _ = cv2.Rodrigues(rvec)
        return R, tvec.reshape(3, 1), count

    def _pnp_verify(self, kp_curr, desc_curr, cand: RelocFrame):
        R, t, count = self._pnp_pose(kp_curr, desc_curr, cand)
        return R, t, count >= self.min_pnp_inliers
