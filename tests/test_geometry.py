import numpy as np
import cv2

from slam.trajectory import Trajectory
from slam.triangulation import Triangulator


def test_world_camera_roundtrip():
    traj = Trajectory()
    rvec = np.array([0.0, 0.0, 0.1])
    R, _ = cv2.Rodrigues(rvec)
    t = np.array([[0.5], [0.0], [0.0]])
    traj.update(R, t)

    R_wc, t_wc = traj.get_latest_Rt()
    T = np.eye(4)
    T[:3, :3] = R_wc
    T[:3, 3] = t_wc.ravel()
    R2, t2 = traj.world_to_camera(traj.get_latest_pose())
    assert np.allclose(R_wc, R2)
    assert np.allclose(t_wc, t2)
    assert np.allclose(T, np.block([[R_wc, t_wc], [np.zeros((1, 3)), np.ones((1, 1))]]))


def test_triangulation_returns_aligned_mask():
    tri = Triangulator()
    pts1 = np.array([[300, 200], [320, 200], [300, 220], [320, 220], [310, 210]], dtype=float)
    pts2 = pts1.copy()
    pts2[:, 0] -= 8.0
    pts, mask = tri.triangulate(
        np.eye(3), np.zeros((3, 1)),
        np.eye(3), np.array([[-0.08], [0.0], [0.0]]),
        pts1, pts2, return_mask=True,
    )
    assert mask.shape == (len(pts1),)
    assert len(pts) == int(mask.sum())
    assert len(pts) >= 4
