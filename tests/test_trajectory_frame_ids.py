import numpy as np

from slam.trajectory import Trajectory


def test_saved_trajectory_preserves_source_frame_ids(tmp_path):
    trajectory = Trajectory()
    trajectory.update(np.eye(3), np.array([[1.0], [0.0], [0.0]]), frame_idx=3)
    trajectory.update(np.eye(3), np.array([[1.0], [0.0], [0.0]]), frame_idx=7)

    positions = tmp_path / "positions.csv"
    poses = tmp_path / "poses.csv"
    trajectory.save_positions_csv(positions)
    trajectory.save_poses_csv(poses)

    positions_rows = np.loadtxt(positions, delimiter=",", skiprows=1)
    poses_rows = np.loadtxt(poses, delimiter=",", skiprows=1)
    assert positions_rows[:, 0].tolist() == [0.0, 3.0, 7.0]
    assert poses_rows[:, 0].tolist() == [0.0, 3.0, 7.0]
