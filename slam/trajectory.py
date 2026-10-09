# trajectory.py - Camera-to-world trajectory

from pathlib import Path
import numpy as np


class Trajectory:
    """Stores camera-to-world poses.

    Relative pose from recoverPose is world/previous-camera -> current-camera:
        X_current = R_rel X_previous + t_rel.
    Therefore the new camera-to-world pose is:
        T_cw_new = T_cw_prev @ inv(T_rel).
    """

    def __init__(self):
        self.poses: list[np.ndarray] = [np.eye(4)]

    @staticmethod
    def world_to_camera(T_cw: np.ndarray):
        T_cw = np.asarray(T_cw, dtype=np.float64).reshape(4, 4)
        R_cw = T_cw[:3, :3]
        C_w = T_cw[:3, 3]
        R_wc = R_cw.T
        t_wc = -R_wc @ C_w
        return R_wc, t_wc.reshape(3, 1)

    def update(self, R_rel, t_rel):
        T_rel = np.eye(4)
        T_rel[:3, :3] = np.asarray(R_rel, dtype=np.float64).reshape(3, 3)
        T_rel[:3, 3] = np.asarray(t_rel, dtype=np.float64).ravel()
        self.poses.append(self.poses[-1] @ np.linalg.inv(T_rel))

    def get_latest_pose(self):
        return self.poses[-1].copy()

    def set_latest_world_to_camera(self, R_wc, t_wc):
        """Replace the latest pose using an absolute world-to-camera estimate."""
        R_wc = np.asarray(R_wc, dtype=np.float64).reshape(3, 3)
        t_wc = np.asarray(t_wc, dtype=np.float64).reshape(3)
        R_cw = R_wc.T
        C_w = -R_cw @ t_wc
        T = np.eye(4)
        T[:3, :3] = R_cw
        T[:3, 3] = C_w
        self.poses[-1] = T

    def get_latest_Rt(self):
        return self.world_to_camera(self.poses[-1])

    def get_positions(self):
        return np.asarray([T[:3, 3] for T in self.poses])

    def apply_correction(self, correction, from_idx):
        for i in range(max(0, from_idx), len(self.poses)):
            self.poses[i] = correction @ self.poses[i]

    def length(self):
        return len(self.poses)

    def save_positions_csv(self, path):
        positions = self.get_positions()
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = np.column_stack([np.arange(len(positions)), positions])
        np.savetxt(out_path, rows, delimiter=",", header="frame_idx,x,y,z", comments="")

    def save_poses_csv(self, path):
        """Persist full camera-to-world poses for rotation-aware evaluation."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for i, T in enumerate(self.poses):
            rows.append(np.concatenate([[i], T[:3, :4].reshape(-1)]))
        np.savetxt(
            out_path, np.asarray(rows), delimiter=",",
            header="frame_idx," + ",".join(f"p{i}" for i in range(12)),
            comments="",
        )
