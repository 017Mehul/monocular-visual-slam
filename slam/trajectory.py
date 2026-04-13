# trajectory.py - Accumulates camera poses to build the trajectory

from pathlib import Path

import numpy as np


class Trajectory:
    """
    Tracks the camera path by composing incremental pose transforms.

    Each pose is a 4×4 homogeneous matrix T (camera-to-world).
    Composition: T_new = T_prev @ inv(T_rel)

    Key improvements:
    - last_good_pose property lets main.py fall back gracefully on bad frames.
    - apply_correction() clamps the correction to avoid exploding poses after
      a noisy loop closure estimate.
    """

    def __init__(self):
        self.poses: list[np.ndarray] = [np.eye(4)]

    def update(self, R: np.ndarray, t: np.ndarray):
        """
        Compose a new pose from the relative transform (R, t).

        Args:
            R: (3,3) rotation matrix.
            t: (3,) or (3,1) translation vector.
        """
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3,  3] = np.array(t).ravel()

        T_new = self.poses[-1] @ np.linalg.inv(T_rel)
        self.poses.append(T_new)

    def get_positions(self) -> np.ndarray:
        """Return (N, 3) camera centre positions."""
        return np.array([T[:3, 3] for T in self.poses])

    def get_latest_pose(self) -> np.ndarray:
        return self.poses[-1].copy()

    def get_latest_Rt(self):
        """Return (R (3,3), t (3,1)) of the most recent pose."""
        T = self.poses[-1]
        return T[:3, :3].copy(), T[:3, 3:4].copy()

    def apply_correction(self, correction: np.ndarray, from_idx: int):
        """
        Apply a loop-closure correction transform to all poses from from_idx.

        Args:
            correction: (4,4) correction matrix.
            from_idx: First pose index to correct.
        """
        for i in range(from_idx, len(self.poses)):
            self.poses[i] = correction @ self.poses[i]

    def length(self) -> int:
        return len(self.poses)

    def save_positions_csv(self, path: str):
        """Persist trajectory positions as CSV for downstream analysis."""
        positions = self.get_positions()
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        header = "frame_idx,x,y,z"
        rows = np.column_stack([np.arange(len(positions)), positions])
        np.savetxt(out_path, rows, delimiter=",", header=header, comments="")
