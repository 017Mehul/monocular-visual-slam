import cv2
import numpy as np

from slam.relocalization import Relocalizer


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
