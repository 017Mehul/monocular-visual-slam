import numpy as np

from scripts.kitti_large_benchmark import aligned_rpe, read_estimated_pose_rows


def test_estimated_pose_csv_preserves_source_frame_ids(tmp_path):
    path = tmp_path / "trajectory_poses.csv"
    identity = np.eye(4)[:3, :4].reshape(-1)
    shifted = np.eye(4)
    shifted[0, 3] = 1.0
    path.write_text(
        "frame_idx," + ",".join(f"p{i}" for i in range(12)) + "\n"
        + "0," + ",".join(map(str, identity)) + "\n"
        + "2," + ",".join(map(str, shifted[:3, :4].reshape(-1))) + "\n",
        encoding="utf-8",
    )
    poses = read_estimated_pose_rows(path)
    assert set(poses) == {0, 2}
    assert poses[2][0, 3] == 1.0


def test_rpe_only_uses_adjacent_frame_ids():
    gt = np.repeat(np.eye(4)[None, :, :], 3, axis=0)
    est = gt.copy()
    # Frames 2 and 3 share the same offset. Comparing frame 0 directly to
    # frame 2 would report error, but that pair is not adjacent and must skip.
    est[1, 0, 3] = 10.0
    est[2, 0, 3] = 10.0
    ids = [0, 2, 3]
    trans, rot = aligned_rpe(gt, est, ids, delta=1)
    assert trans == 0.0
    assert rot == 0.0
