# scale_estimator.py - Explicit monocular scale policy


class ScaleEstimator:
    """Manage scale for monocular VO.

    A monocular camera cannot observe metric translation scale from two frames
    alone. The old implementation attempted to infer scale from unrelated
    depth arrays, which silently returned stale values. This class therefore
    makes the limitation explicit and supports an optional externally supplied
    metric scale (e.g. wheel/IMU/GNSS or a known baseline).
    """

    def __init__(self, initial_scale: float = 1.0):
        if initial_scale <= 0:
            raise ValueError("initial_scale must be positive")
        self._scale = float(initial_scale)

    @property
    def scale(self):
        return self._scale

    def set_scale(self, scale: float):
        if scale <= 0:
            raise ValueError("scale must be positive")
        self._scale = float(scale)

    def estimate(self, *_args, **_kwargs):
        """Return the configured scale; no fake depth-based inference."""
        return self._scale

    def reset(self):
        self._scale = 1.0
