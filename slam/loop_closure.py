# loop_closure.py - Loop closure detection with pose graph correction

import cv2
import numpy as np
from slam.config import LOOP_CLOSURE_PARAMS, CAMERA_PARAMS, RANSAC_PARAMS


class Keyframe:
    def __init__(self, frame_idx, pose_idx, keypoints, descriptors):
        self.frame_idx = frame_idx
        self.pose_idx = pose_idx
        self.keypoints = keypoints
        self.descriptors = descriptors.astype(np.uint8)
        self._mean = descriptors.astype(np.float32).mean(axis=0)


class LoopClosureDetector:
    def __init__(self):
        self.keyframes: list[Keyframe] = []
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        cfg = LOOP_CLOSURE_PARAMS
        self.min_interval = cfg["min_keyframe_interval"]
        self.sim_threshold = cfg["similarity_threshold"]
        self.min_matches = cfg["min_matches"]
        self.min_skip = cfg["min_skip_frames"]
        self._last_kf_idx = -self.min_interval

        p = CAMERA_PARAMS
        self.K = np.array([
            [p["fx"],      0, p["cx"]],
            [     0, p["fy"], p["cy"]],
            [     0,       0,       1],
        ], dtype=np.float64)

    def register_keyframe(self, frame_idx, pose_idx, keypoints, descriptors):
        if descriptors is None or len(descriptors) < 10:
            return
        if frame_idx - self._last_kf_idx < self.min_interval:
            return
        self.keyframes.append(Keyframe(frame_idx, pose_idx, keypoints, descriptors))
        self._last_kf_idx = frame_idx

    def detect(self, frame_idx, keypoints, descriptors):
        if descriptors is None or len(self.keyframes) < 3:
            return False, None, None, None

        candidates = [kf for kf in self.keyframes if frame_idx - kf.frame_idx > self.min_skip]
        if not candidates:
            return False, None, None, None

        curr_mean = descriptors.astype(np.float32).mean(axis=0)
        best_kf, best_sim = self._best_candidate(curr_mean, candidates)

        if best_sim < self.sim_threshold:
            return False, None, None, None

        R, t, n_inliers = self._pnp_verify(keypoints, descriptors, best_kf)
        if n_inliers < self.min_matches:
            return False, None, None, None

        print(f"[LoopClosure] frame {frame_idx} → kf {best_kf.frame_idx} (sim={best_sim:.3f}, pnp_inliers={n_inliers})")
        return True, best_kf, R, t

    def compute_correction(self, current_pose, loop_pose):
        return loop_pose @ np.linalg.inv(current_pose)

    def apply_smooth_correction(self, poses, from_idx, correction):
        n = len(poses) - from_idx
        if n <= 0:
            return

        R_c = correction[:3, :3]
        t_c = correction[:3, 3]

        angle = np.arccos(np.clip((np.trace(R_c) - 1) / 2, -1, 1))
        if angle < 1e-8:
            rvec_c = np.zeros(3)
        else:
            k = np.array([R_c[2,1]-R_c[1,2], R_c[0,2]-R_c[2,0], R_c[1,0]-R_c[0,1]]) / (2 * np.sin(angle))
            rvec_c = k * angle

        for i, idx in enumerate(range(from_idx, len(poses))):
            alpha = (i + 1) / n
            rvec_i = rvec_c * alpha
            theta = np.linalg.norm(rvec_i)
            if theta < 1e-8:
                R_i = np.eye(3)
            else:
                k = rvec_i / theta
                K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
                R_i = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)
            t_i = t_c * alpha
            T_i = np.eye(4)
            T_i[:3, :3] = R_i
            T_i[:3,  3] = t_i
            poses[idx] = T_i @ poses[idx]

    def _best_candidate(self, curr_mean, candidates):
        best_sim, best_kf = -1.0, None
        for kf in candidates:
            d = np.linalg.norm(curr_mean) * np.linalg.norm(kf._mean)
            sim = float(np.dot(curr_mean, kf._mean) / d) if d > 1e-8 else 0.0
            if sim > best_sim:
                best_sim, best_kf = sim, kf
        return best_kf, best_sim

    def _pnp_verify(self, kp_curr, desc_curr, kf: Keyframe):
        raw = self.matcher.knnMatch(desc_curr, kf.descriptors, k=2)
        good = [m for m, n in raw if len([m, n]) == 2 and m.distance < 0.75 * n.distance]
        if len(good) < 8:
            return None, None, 0
        pts_2d = np.float64([kp_curr[m.queryIdx].pt for m in good])
        pts_kf = np.float64([kf.keypoints[m.trainIdx].pt for m in good])
        E, mask = cv2.findEssentialMat(pts_2d, pts_kf, self.K, method=cv2.RANSAC, prob=0.999, threshold=RANSAC_PARAMS["threshold"])
        if E is None or mask is None:
            return None, None, 0
        n_inliers, R, t, _ = cv2.recoverPose(E, pts_2d, pts_kf, self.K, mask=mask)
        return R, t.reshape(3, 1), n_inliers
