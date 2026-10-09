# keyframe_manager.py - Keyframe selection with consistent 2D/3D observations

import numpy as np


class Keyframe:
    """Keyframe with explicitly associated 2D observations and 3D landmarks."""

    _counter = 0

    def __init__(
        self, frame_idx: int, R: np.ndarray, t: np.ndarray,
        keypoints, descriptors, points_3d: np.ndarray,
    ):
        self.id = Keyframe._counter
        Keyframe._counter += 1
        self.frame_idx = frame_idx
        self.R = np.asarray(R, dtype=np.float64).copy()
        self.t = np.asarray(t, dtype=np.float64).reshape(3, 1).copy()
        self.keypoints = list(keypoints)
        self.descriptors = descriptors.copy() if descriptors is not None else None
        self.points_3d = (
            np.asarray(points_3d, dtype=np.float64).reshape(-1, 3).copy()
            if points_3d is not None else np.empty((0, 3))
        )
        # One observation per landmark: (landmark index, u, v).
        n = min(len(self.keypoints), len(self.points_3d))
        self.observations = [
            (i, float(self.keypoints[i].pt[0]), float(self.keypoints[i].pt[1]))
            for i in range(n)
        ]


class KeyframeManager:
    """Maintains a sliding window for local bundle adjustment."""

    def __init__(
        self, min_frames: int = 5, min_baseline: float = 0.02,
        max_overlap: float = 0.90, window_size: int = 7,
    ):
        self.min_frames = min_frames
        self.min_baseline = min_baseline
        self.max_overlap = max_overlap
        self.window_size = window_size
        self.keyframes: list[Keyframe] = []
        self._last_kf_frame_idx = -min_frames

    def should_insert(self, frame_idx, R, t, n_matches, n_prev_features):
        if not self.keyframes:
            return True
        if frame_idx - self._last_kf_frame_idx < self.min_frames:
            return False
        if np.linalg.norm(np.asarray(t).ravel()) < self.min_baseline:
            return False
        overlap = n_matches / max(n_prev_features, 1)
        return overlap <= self.max_overlap

    def insert(self, frame_idx, R, t, keypoints, descriptors, points_3d):
        kf = Keyframe(frame_idx, R, t, keypoints, descriptors, points_3d)
        self.keyframes.append(kf)
        self._last_kf_frame_idx = frame_idx
        if len(self.keyframes) > self.window_size:
            self.keyframes.pop(0)
        return kf

    def get_window_poses(self):
        return [(kf.R, kf.t) for kf in self.keyframes]

    def build_observations(self):
        """Build cross-keyframe BA observations using descriptor associations.

        Each keyframe contributes its own landmarks to the global point vector.
        The same landmark can then be observed by other keyframes when their
        descriptors pass a ratio-tested Hamming match. This gives BA actual
        multi-view constraints instead of optimizing independent points.
        """
        observations = []
        all_pts = []
        blocks = []
        offset = 0

        for kf in self.keyframes:
            n = len(kf.points_3d)
            blocks.append((offset, n))
            all_pts.extend(kf.points_3d.tolist())
            offset += n

        pts_array = np.asarray(all_pts, dtype=np.float64).reshape(-1, 3)
        if len(self.keyframes) == 0:
            return [], pts_array

        # Own-keyframe observations are exact by construction.
        for cam_idx, kf in enumerate(self.keyframes):
            base, n = blocks[cam_idx]
            for local_idx, u, v in kf.observations:
                if local_idx < n:
                    observations.append((cam_idx, base + local_idx, u, v))

        # Cross-keyframe associations: landmark descriptor -> observing frame.
        import cv2
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        seen = {(cam, pt) for cam, pt, _, _ in observations}

        for src_idx, src_kf in enumerate(self.keyframes):
            if src_kf.descriptors is None or len(src_kf.descriptors) == 0:
                continue
            src_base, src_n = blocks[src_idx]
            src_n = min(src_n, len(src_kf.descriptors))
            if src_n == 0:
                continue

            for cam_idx, dst_kf in enumerate(self.keyframes):
                if cam_idx == src_idx or dst_kf.descriptors is None or len(dst_kf.descriptors) < 2:
                    continue
                raw = matcher.knnMatch(src_kf.descriptors[:src_n], dst_kf.descriptors, k=2)
                for pair in raw:
                    if len(pair) != 2:
                        continue
                    m, n = pair
                    if m.distance >= 0.75 * n.distance:
                        continue
                    pt_idx = src_base + m.queryIdx
                    key = (cam_idx, pt_idx)
                    if key in seen:
                        continue
                    if m.trainIdx < len(dst_kf.keypoints):
                        u, v = dst_kf.keypoints[m.trainIdx].pt
                        observations.append((cam_idx, pt_idx, float(u), float(v)))
                        seen.add(key)

        return observations, pts_array

    def update_poses_from_ba(self, optimised_poses):
        for kf, (R, t) in zip(self.keyframes, optimised_poses):
            kf.R = np.asarray(R, dtype=np.float64).copy()
            kf.t = np.asarray(t, dtype=np.float64).reshape(3, 1).copy()

    def update_points_from_ba(self, optimised_points):
        pts = np.asarray(optimised_points, dtype=np.float64).reshape(-1, 3)
        offset = 0
        for kf in self.keyframes:
            n = len(kf.points_3d)
            if offset + n <= len(pts):
                kf.points_3d = pts[offset:offset + n].copy()
            offset += n

    def get_window_points(self):
        """Return all current local-window landmarks in BA ordering."""
        if not self.keyframes:
            return np.empty((0, 3), dtype=np.float64)
        return np.vstack([kf.points_3d for kf in self.keyframes]).astype(np.float64, copy=False)

    def size(self):
        return len(self.keyframes)

    def last_keyframe(self):
        return self.keyframes[-1] if self.keyframes else None
