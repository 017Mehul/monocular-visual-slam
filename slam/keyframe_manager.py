# keyframe_manager.py - Keyframe selection with baseline and overlap checks

import numpy as np


class Keyframe:
    """One keyframe entry in the sliding window."""

    _counter = 0

    def __init__(self, frame_idx: int, R: np.ndarray, t: np.ndarray,
                 keypoints, descriptors, points_3d: np.ndarray):
        self.id = Keyframe._counter
        Keyframe._counter += 1
        self.frame_idx = frame_idx
        self.R = R.copy()
        self.t = t.copy().reshape(3, 1)
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.points_3d = points_3d.copy() if points_3d is not None else np.empty((0, 3))
        # Observations: list of (pt_idx_in_window, u, v)
        self.observations: list[tuple] = []


class KeyframeManager:
    """
    Decides when to insert keyframes and maintains a sliding window
    of recent keyframes for local Bundle Adjustment.
    """

    def __init__(
        self,
        min_frames: int = 5,
        min_baseline: float = 0.02,   # fraction of median scene depth
        max_overlap: float = 0.90,    # max ratio of shared features
        window_size: int = 7,
    ):
        self.min_frames = min_frames
        self.min_baseline = min_baseline
        self.max_overlap = max_overlap
        self.window_size = window_size

        self.keyframes: list[Keyframe] = []
        self._last_kf_frame_idx = -min_frames

    def should_insert(
        self,
        frame_idx: int,
        R: np.ndarray,
        t: np.ndarray,
        n_matches: int,
        n_prev_features: int,
    ) -> bool:
        if not self.keyframes:
            return True
        if frame_idx - self._last_kf_frame_idx < self.min_frames:
            return False
        baseline = np.linalg.norm(t)
        if baseline < self.min_baseline:
            return False
        overlap = n_matches / max(n_prev_features, 1)
        if overlap > self.max_overlap:
            return False
        return True

    def insert(
        self,
        frame_idx: int,
        R: np.ndarray,
        t: np.ndarray,
        keypoints,
        descriptors,
        points_3d: np.ndarray,
    ) -> Keyframe:
        kf = Keyframe(frame_idx, R, t, keypoints, descriptors, points_3d)
        self.keyframes.append(kf)
        self._last_kf_frame_idx = frame_idx
        if len(self.keyframes) > self.window_size:
            self.keyframes.pop(0)
        return kf

    def get_window_poses(self):
        return [(kf.R, kf.t) for kf in self.keyframes]

    def get_window_points(self) -> np.ndarray:
        all_pts = [kf.points_3d for kf in self.keyframes
                   if len(kf.points_3d) > 0]
        if not all_pts:
            return np.empty((0, 3))
        return np.vstack(all_pts)

    def build_observations(self):
        observations = []
        all_pts = []
        pt_offset = 0

        for cam_idx, kf in enumerate(self.keyframes):
            for i, pt in enumerate(kf.points_3d):
                all_pts.append(pt)
                if i < len(kf.keypoints):
                    u, v = kf.keypoints[i].pt
                    observations.append((cam_idx, pt_offset + i, u, v))
            pt_offset += len(kf.points_3d)

        pts_array = np.array(all_pts) if all_pts else np.empty((0, 3))
        return observations, pts_array

    def update_poses_from_ba(self, optimised_poses):
        for kf, (R, t) in zip(self.keyframes, optimised_poses):
            kf.R = R
            kf.t = t.reshape(3, 1)

    def size(self) -> int:
        return len(self.keyframes)

    def last_keyframe(self):
        return self.keyframes[-1] if self.keyframes else None
