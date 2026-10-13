import numpy as np

from slam.keyframe_manager import KeyframeManager
from slam.loop_closure import LoopClosureDetector


class Point:
    def __init__(self, u, v):
        self.pt = (u, v)


def test_loop_correction_returns_transforms_for_multi_pose_trajectory():
    detector = LoopClosureDetector()
    poses = [np.eye(4) for _ in range(4)]
    for i, pose in enumerate(poses):
        pose[0, 3] = float(i)
    before = [p.copy() for p in poses]
    correction = np.eye(4)
    correction[1, 3] = 2.0

    per_pose = detector.apply_smooth_correction(poses, 1, correction)

    assert set(per_pose) == {1, 2, 3}
    assert np.allclose(poses[0], before[0])
    assert np.allclose(poses[-1], per_pose[3] @ before[-1])
    assert all(np.isfinite(p).all() for p in poses)


def test_loop_correction_updates_keyframe_poses_and_landmarks_together():
    manager = KeyframeManager(min_frames=1)
    points = np.array([[1.0, 0.0, 5.0], [2.0, 1.0, 6.0], [3.0, 2.0, 7.0], [4.0, 3.0, 8.0]])
    descriptors = np.zeros((4, 32), dtype=np.uint8)
    manager.insert(5, np.eye(3), np.zeros((3, 1)), [Point(i, i) for i in range(4)], descriptors, points)
    C = np.eye(4)
    C[0, 3] = 2.0

    manager.apply_world_corrections({5: C})

    assert np.allclose(manager.keyframes[0].points_3d[:, 0], points[:, 0] + 2.0)
    T_wc = np.eye(4)
    T_wc[:3, :3] = manager.keyframes[0].R
    T_wc[:3, 3] = manager.keyframes[0].t.ravel()
    assert np.allclose(T_wc @ C, np.eye(4))
