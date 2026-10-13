import numpy as np

from slam.pose_safeguards import (
    is_valid_rotation,
    is_valid_world_to_camera_pose,
    translation_update_is_plausible,
)


def test_pose_safeguards_accept_proper_rotation_and_finite_translation():
    assert is_valid_rotation(np.eye(3))
    assert is_valid_world_to_camera_pose(np.eye(3), [1.0, -2.0, 0.5])


def test_pose_safeguards_reject_nonfinite_and_nonrigid_pose():
    assert not is_valid_rotation(np.diag([1.0, 1.0, 2.0]))
    assert not is_valid_rotation(np.full((3, 3), np.nan))
    assert not is_valid_world_to_camera_pose(np.eye(3), [0.0, np.inf, 0.0])


def test_translation_gate_rejects_isolated_huge_jump():
    history = np.column_stack([np.arange(10.0), np.zeros(10), np.zeros(10)])
    accepted, error, limit = translation_update_is_plausible(
        [10.0, 0.0, 0.0], [900.0, 0.0, 0.0], history
    )
    assert not accepted
    assert error > limit


def test_translation_gate_is_adaptive_to_consistently_fast_motion():
    history = np.column_stack([np.arange(10.0) * 20.0, np.zeros(10), np.zeros(10)])
    accepted, error, limit = translation_update_is_plausible(
        [200.0, 0.0, 0.0], [201.0, 0.0, 0.0], history
    )
    assert accepted
    assert error <= limit
