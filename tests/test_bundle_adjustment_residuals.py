import numpy as np

from slam.bundle_adjustment import _R_to_rodrigues, _project, _residuals, _pack
from slam.pose_estimation import build_intrinsic_matrix


def test_vectorized_residuals_match_scalar_projection():
    K = build_intrinsic_matrix()
    camera_params = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.02, 0.0, 0.3, -0.1, 0.2],
    ])
    points = np.array([
        [0.1, 0.2, 4.0],
        [-0.5, 0.1, 6.0],
        [0.3, -0.2, 5.0],
    ])
    cam_indices = np.array([0, 1, 1])
    pt_indices = np.array([0, 1, 2])
    obs = np.array([[600.0, 220.0], [550.0, 190.0], [620.0, 160.0]])
    x = _pack(camera_params, points)

    actual = _residuals(x, 2, 3, cam_indices, pt_indices, obs, K)
    expected = np.concatenate([
        _project(K, camera_params[c, :3], camera_params[c, 3:], points[p]) - o
        for c, p, o in zip(cam_indices, pt_indices, obs)
    ])

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)


def test_vectorized_residuals_preserve_behind_camera_behavior():
    K = build_intrinsic_matrix()
    camera_params = np.zeros((1, 6))
    points = np.array([[0.0, 0.0, -1.0]])
    obs = np.array([[10.0, 20.0]])
    actual = _residuals(
        _pack(camera_params, points), 1, 1,
        np.array([0]), np.array([0]), obs, K,
    )
    np.testing.assert_allclose(actual, [-10.0, -20.0])
