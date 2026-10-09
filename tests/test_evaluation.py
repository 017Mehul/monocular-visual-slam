import numpy as np

from slam.kitti_evaluation import compute_ate, compute_rpe


def test_ate_similarity_alignment_zero_for_equivalent_scale():
    gt = np.array([[0,0,0], [1,0,0], [2,0,0], [3,0,0]], dtype=float)
    est = gt * 2.0
    assert compute_ate(gt, est) < 1e-8


def test_rpe_identity_poses_zero():
    poses = np.eye(4)[None, :, :].repeat(5, axis=0)
    assert compute_rpe(poses, poses) == (0.0, 0.0)
