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
        """Return BA observations with local landmark indices.

        Each keyframe owns its landmark block, so point indices are guaranteed
        to match the corresponding keypoints.
        """
        observations = []
        all_pts = []
        offset = 0
        for cam_idx, kf in enumerate(self.keyframes):
            all_pts.extend(kf.points_3d.tolist())
            for local_idx, u, v in kf.observations:
                if local_idx < len(kf.points_3d):
                    observations.append((cam_idx, offset + local_idx, u, v))
            offset += len(kf.points_3d)
        pts = np.asarray(all_pts, dtype=np.float64).reshape(-1, 3)
        return observations, pts

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

    def size(self):
        return len(self.keyframes)

    def last_keyframe(self):
        return self.keyframes[-1] if self.keyframes else None
