"""Geometric sanity checks for pose updates from independent SLAM stages.

These checks reject numerically invalid rigid transforms and isolated translation
outliers. They deliberately use recent trajectory motion rather than a fixed
per-frame speed cap, so a consistently fast segment is not rejected merely for
being fast.
"""
from __future__ import annotations

import numpy as np


def is_valid_rotation(rotation, atol=5e-2):
    """Return whether a matrix is a finite, approximately proper SO(3) rotation."""
    R = np.asarray(rotation, dtype=np.float64)
    if R.shape != (3, 3) or not np.isfinite(R).all():
        return False
    if not np.allclose(R.T @ R, np.eye(3), atol=atol, rtol=0.0):
        return False
    determinant = float(np.linalg.det(R))
    return bool(np.isfinite(determinant) and abs(determinant - 1.0) <= atol)


def is_valid_world_to_camera_pose(rotation, translation):
    """Validate finite translation and a proper world-to-camera rotation."""
    t = np.asarray(translation, dtype=np.float64).reshape(-1)
    return bool(t.shape == (3,) and np.isfinite(t).all()
                and is_valid_rotation(rotation))


def recent_motion_scale(positions, window=20):
    """Robust typical step length from camera-to-world positions."""
    points = np.asarray(positions, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        return 1.0
    points = points[-(max(2, int(window)) + 1):]
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    steps = steps[np.isfinite(steps) & (steps > 1e-9)]
    if not len(steps):
        return 1.0
    return float(np.median(steps))


def translation_update_is_plausible(reference_position, candidate_position,
                                    recent_positions, max_median_steps=8.0,
                                    absolute_floor=3.0):
    """Reject isolated pose jumps while adapting to the sequence's motion scale.

    The candidate is compared with the independent VO prediction, not simply
    with the previous frame. The tolerance is a robust multiple of recent
    motion with a small floor for low-motion/bootstrap segments. No absolute
    metric speed assumption is made.
    """
    reference = np.asarray(reference_position, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate_position, dtype=np.float64).reshape(-1)
    if reference.shape != (3,) or candidate.shape != (3,):
        return False, float("inf"), float("nan")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        return False, float("inf"), float("nan")
    error = float(np.linalg.norm(candidate - reference))
    scale = recent_motion_scale(recent_positions)
    limit = max(float(absolute_floor), float(max_median_steps) * scale)
    return bool(np.isfinite(error) and error <= limit), error, limit
