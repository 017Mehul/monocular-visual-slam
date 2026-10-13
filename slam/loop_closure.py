# loop_closure.py - Feature-based loop candidate verification

import cv2
import numpy as np
from slam.bow_loop_closure import BoWLoopClosure
from slam.config import LOOP_CLOSURE_PARAMS, CAMERA_PARAMS, RANSAC_PARAMS


class Keyframe:
    def __init__(self, frame_idx, pose_idx, keypoints, descriptors):
        self.frame_idx = frame_idx
        self.pose_idx = pose_idx
        self.keypoints = keypoints
        self.descriptors = descriptors.astype(np.uint8)


class LoopClosureDetector:
    """Detect visually similar keyframes and geometrically verify them.

    Verification uses an Essential matrix because this detector stores 2D
    keyframes only. It is deliberately named essential_verify, not PnP.
    """

    def __init__(self):
        self.keyframes = []
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        cfg = LOOP_CLOSURE_PARAMS
        self.min_interval = cfg["min_keyframe_interval"]
        self.sim_threshold = cfg["similarity_threshold"]
        self.min_matches = cfg["min_matches"]
        self.min_skip = cfg["min_skip_frames"]
        self.candidate_top_k = int(cfg.get("candidate_top_k", 10))
        self._last_kf_idx = -self.min_interval
        self.bow = BoWLoopClosure(words=cfg.get("bow_words", 256), min_interval=self.min_interval, min_skip_frames=self.min_skip)
        p = CAMERA_PARAMS
        self.K = np.array([[p["fx"], 0, p["cx"]], [0, p["fy"], p["cy"]], [0, 0, 1]], dtype=np.float64)

    def register_keyframe(self, frame_idx, pose_idx, keypoints, descriptors):
        if descriptors is None or len(descriptors) < 10:
            return
        if frame_idx - self._last_kf_idx < self.min_interval:
            return
        kf = Keyframe(frame_idx, pose_idx, keypoints, descriptors)
        self.keyframes.append(kf)
        self.bow.register(frame_idx, kf, descriptors)
        self._last_kf_idx = frame_idx

    def detect(self, frame_idx, keypoints, descriptors):
        if descriptors is None or len(self.keyframes) < 3:
            return False, None, None, None
        candidates = [kf for kf in self.keyframes if frame_idx - kf.frame_idx > self.min_skip]
        if not candidates:
            return False, None, None, None
        bow_candidates = self.bow.query(frame_idx, descriptors, top_k=self.candidate_top_k)
        if not bow_candidates:
            # Do not silently fall back to descriptor-mean similarity: it is
            # not a reliable place-recognition score and can create false loops.
            return False, None, None, None
        # BoW is a retrieval stage, not a final decision. A visually similar
        # first candidate may fail geometric verification while another of the
        # top-ranked candidates is the genuine revisit. Try the ranked shortlist
        # in score order and accept only a geometrically verified candidate.
        for similarity, candidate in bow_candidates:
            if candidate is None or similarity < self.sim_threshold:
                break
            R, t, n_inliers = self._essential_verify(keypoints, descriptors, candidate)
            if R is not None and t is not None and n_inliers >= self.min_matches:
                return True, candidate, R, t
        return False, None, None, None

    @staticmethod
    def pose_consistent(current_pose, candidate_pose, relative_R, relative_t,
                        max_rotation_error_deg=20.0,
                        min_translation_direction_cosine=0.5):
        """Reject geometrically verified matches that contradict the VO pose prior.

        Essential-matrix verification alone can accept repeated textures. Compare
        its relative rotation and translation direction with the current trajectory
        before allowing a loop correction. Translation magnitude is deliberately
        ignored because monocular essential geometry has unknown scale.
        """
        current_pose = np.asarray(current_pose, dtype=np.float64).reshape(4, 4)
        candidate_pose = np.asarray(candidate_pose, dtype=np.float64).reshape(4, 4)
        relative_R = np.asarray(relative_R, dtype=np.float64).reshape(3, 3)
        relative_t = np.asarray(relative_t, dtype=np.float64).reshape(3)

        expected_R = candidate_pose[:3, :3].T @ current_pose[:3, :3]
        rotation_error = relative_R @ expected_R.T
        cos_angle = np.clip((np.trace(rotation_error) - 1.0) / 2.0, -1.0, 1.0)
        angle_deg = float(np.degrees(np.arccos(cos_angle)))
        if not np.isfinite(angle_deg) or angle_deg > max_rotation_error_deg:
            return False

        expected_t = candidate_pose[:3, :3].T @ (
            current_pose[:3, 3] - candidate_pose[:3, 3]
        )
        expected_norm = float(np.linalg.norm(expected_t))
        measured_norm = float(np.linalg.norm(relative_t))
        # If the current trajectory has essentially zero baseline, rotation is
        # the only meaningful consistency signal available.
        if expected_norm < 1e-6 or measured_norm < 1e-6:
            return True
        cosine = float(np.dot(expected_t, relative_t) / (expected_norm * measured_norm))
        return bool(np.isfinite(cosine) and cosine >= min_translation_direction_cosine)

    def compute_correction(self, current_pose, loop_pose):
        return np.asarray(loop_pose) @ np.linalg.inv(np.asarray(current_pose))

    def apply_smooth_correction(self, poses, from_idx, correction):
        """Apply a smoothly varying world-frame correction and return per-pose transforms."""
        n = len(poses) - from_idx
        if n <= 0:
            return {}
        original_poses = {idx: np.asarray(poses[idx]).copy() for idx in range(from_idx, len(poses))}
        R_c = correction[:3, :3]
        t_c = correction[:3, 3]
        angle = np.arccos(np.clip((np.trace(R_c) - 1) / 2, -1, 1))
        if angle < 1e-8:
            rvec_c = np.zeros(3)
        else:
            denom = max(2 * np.sin(angle), 1e-8)
            axis = np.array([R_c[2, 1] - R_c[1, 2], R_c[0, 2] - R_c[2, 0], R_c[1, 0] - R_c[0, 1]]) / denom
            rvec_c = axis * angle
        for i, idx in enumerate(range(from_idx, len(poses))):
            alpha = (i + 1) / n
            r = rvec_c * alpha
            theta = np.linalg.norm(r)
            if theta < 1e-8:
                R_i = np.eye(3)
            else:
                k = r / theta
                K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
                R_i = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
            T = np.eye(4)
            T[:3, :3] = R_i
            T[:3, 3] = t_c * alpha
            poses[idx] = T @ poses[idx]
        return {
            idx: np.asarray(poses[idx]) @ np.linalg.inv(original_pose)
            for idx, original_pose in original_poses.items()
        }

    def _essential_verify(self, kp_curr, desc_curr, kf):
        raw = self.matcher.knnMatch(desc_curr, kf.descriptors, k=2)
        good = [m for pair in raw if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            return None, None, 0
        pts_curr = np.float64([kp_curr[m.queryIdx].pt for m in good])
        pts_kf = np.float64([kf.keypoints[m.trainIdx].pt for m in good])
        E, mask = cv2.findEssentialMat(
            pts_curr, pts_kf, self.K, method=cv2.RANSAC,
            prob=RANSAC_PARAMS["prob"], threshold=RANSAC_PARAMS["threshold"],
        )
        if E is None or mask is None:
            return None, None, 0
        if E.shape != (3, 3):
            E = E[:3, :3]
        n_inliers, R, t, _ = cv2.recoverPose(E, pts_curr, pts_kf, self.K, mask=mask)
        return R, t.reshape(3, 1), int(n_inliers)
