import cv2
import numpy as np

from slam.keyframe_manager import KeyframeManager


def _kp(x, y):
    return cv2.KeyPoint(float(x), float(y), 8)


def test_keyframe_observations_keep_landmark_alignment():
    mgr = KeyframeManager(min_frames=1, min_baseline=0.0)
    desc = np.arange(8 * 32, dtype=np.uint8).reshape(8, 32)
    pts = np.array([[1, 2, 8], [2, 3, 9], [3, 4, 10], [4, 5, 11]], dtype=float)
    kps = [_kp(10, 20), _kp(20, 30), _kp(30, 40), _kp(40, 50)]
    mgr.insert(1, np.eye(3), np.zeros((3, 1)), kps, desc[:4], pts)
    obs, all_pts = mgr.build_observations()
    assert all_pts.shape == (4, 3)
    assert len(obs) == 4
    assert all(o[1] < len(all_pts) for o in obs)
