import cv2
import numpy as np

from slam.relocalization import Relocalizer
from slam.keyframe_manager import KeyframeManager


def test_relocalization_ranks_by_real_orb_correspondences():
    rng = np.random.default_rng(20261013)
    descriptors = rng.integers(0, 256, size=(40, 32), dtype=np.uint8)
    unrelated = rng.integers(0, 256, size=(40, 32), dtype=np.uint8)
    keypoints = [cv2.KeyPoint(float(i), float(i % 7), 3) for i in range(40)]
    points = rng.normal(size=(40, 3))

    relocalizer = Relocalizer(min_pnp_inliers=12)
    relocalizer.add_frame(10, keypoints, unrelated, points, np.eye(3), np.zeros((3, 1)))
    relocalizer.add_frame(20, keypoints, descriptors, points, np.eye(3), np.zeros((3, 1)))

    ranked = relocalizer._get_candidates(descriptors, top_k=2)

    assert ranked
    assert ranked[0].frame_idx == 20



def test_relocalization_database_syncs_bundle_adjusted_keyframes():
    descriptors = np.arange(8 * 32, dtype=np.uint8).reshape(8, 32)
    keypoints = [cv2.KeyPoint(float(i), float(i + 1), 3) for i in range(8)]
    points = np.arange(24, dtype=np.float64).reshape(8, 3)

    relocalizer = Relocalizer()
    relocalizer.add_frame(20, keypoints, descriptors, points, np.eye(3), np.zeros((3, 1)))

    manager = KeyframeManager(min_frames=1, min_baseline=0.0)
    kf = manager.insert(20, np.eye(3), np.array([[1.0], [2.0], [3.0]]),
                        keypoints, descriptors, points + 5.0)
    relocalizer.sync_keyframes(manager.keyframes)

    np.testing.assert_allclose(relocalizer._db[0].points_3d, points + 5.0)
    np.testing.assert_allclose(relocalizer._db[0].t.ravel(), [1.0, 2.0, 3.0])
    assert relocalizer._db[0].frame_idx == kf.frame_idx


def test_relocalization_database_applies_loop_world_correction():
    descriptors = np.arange(4 * 32, dtype=np.uint8).reshape(4, 32)
    keypoints = [cv2.KeyPoint(float(i), float(i + 1), 3) for i in range(4)]
    points = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0],
                       [0.0, 1.0, 3.0], [1.0, 1.0, 4.0]])
    relocalizer = Relocalizer()
    relocalizer.add_frame(20, keypoints, descriptors, points,
                          np.eye(3), np.zeros((3, 1)))

    correction = np.eye(4)
    correction[:3, 3] = [2.0, -1.0, 0.5]
    relocalizer.apply_world_corrections({20: correction})

    np.testing.assert_allclose(
        relocalizer._db[0].points_3d,
        points + np.array([2.0, -1.0, 0.5]),
    )
    np.testing.assert_allclose(
        relocalizer._db[0].t.ravel(),
        [-2.0, 1.0, -0.5],
    )



def test_local_map_pnp_recovers_absolute_pose_from_landmarks():
    rng = np.random.default_rng(20261013)
    n = 60
    descriptors = rng.integers(0, 256, size=(n, 32), dtype=np.uint8)
    points = np.column_stack([
        rng.uniform(-2.0, 2.0, n),
        rng.uniform(-1.2, 1.2, n),
        rng.uniform(5.0, 12.0, n),
    ])
    relocalizer = Relocalizer(min_pnp_inliers=12)
    K = relocalizer.K
    rvec = np.array([0.015, -0.025, 0.01], dtype=np.float64)
    R, _ = cv2.Rodrigues(rvec)
    t = np.array([[0.25], [-0.12], [0.35]], dtype=np.float64)
    camera_points = (R @ points.T + t).T
    projected = np.column_stack([
        K[0, 0] * camera_points[:, 0] / camera_points[:, 2] + K[0, 2],
        K[1, 1] * camera_points[:, 1] / camera_points[:, 2] + K[1, 2],
    ])
    keypoints = [cv2.KeyPoint(float(u), float(v), 3) for u, v in projected]
    relocalizer.add_frame(
        10, keypoints, descriptors, points, np.eye(3), np.zeros((3, 1))
    )

    recovered_R, recovered_t, inliers = relocalizer.estimate_pose_from_map(
        keypoints, descriptors, top_k=1
    )

    assert recovered_R is not None
    assert inliers >= 12
    np.testing.assert_allclose(recovered_R, R, atol=1e-2)
    np.testing.assert_allclose(recovered_t, t, atol=1e-1)
