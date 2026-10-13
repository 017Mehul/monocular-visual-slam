import numpy as np

from slam.pose_safeguards import (
    bundle_adjustment_baselines_are_plausible,
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


def test_translation_gate_scale_is_not_inflated_by_recent_outliers():
    # A mostly one-unit trajectory with a few corrupt jumps must keep a small
    # innovation threshold instead of adapting to the corrupt values.
    x = np.array([0, 1, 2, 3, 4, 20, 21, 22, 45, 46, 47], dtype=float)
    history = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    accepted, error, limit = translation_update_is_plausible(
        [48.0, 0.0, 0.0], [55.0, 0.0, 0.0], history
    )
    assert not accepted
    assert error > limit

def test_bundle_adjustment_guard_rejects_isolated_baseline_warp():
    before = np.array([20.0, 20.0, 20.0, 20.0])
    after = np.array([20.0, 38.0, 20.0, 20.0])
    accepted, bad_edges, deviation, tolerance = bundle_adjustment_baselines_are_plausible(
        before, after, typical_step=1.0
    )
    assert not accepted
    assert bad_edges.tolist() == [1]
    assert deviation[1] == 18.0
    assert tolerance[1] == 10.0


def test_bundle_adjustment_guard_allows_small_refinement_and_fast_motion():
    before = np.array([20.0, 25.0, 30.0])
    after = np.array([21.0, 23.0, 32.0])
    accepted, bad_edges, _, _ = bundle_adjustment_baselines_are_plausible(
        before, after, typical_step=4.0
    )
    assert accepted
    assert bad_edges.size == 0

