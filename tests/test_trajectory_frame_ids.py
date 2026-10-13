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


def test_bundle_adjustment_world_to_camera_pose_is_converted_to_camera_to_world():
    trajectory = Trajectory()
    R_wc = np.eye(3)
    t_wc = np.array([[-2.0], [0.0], [0.0]])

    trajectory.set_pose_from_world_to_camera(0, R_wc, t_wc)

    assert trajectory.poses[0][0, 3] == 2.0
    recovered_R, recovered_t = trajectory.world_to_camera(trajectory.poses[0])
    assert np.allclose(recovered_R, R_wc)
    assert np.allclose(recovered_t, t_wc)
