from slam.scale_estimator import ScaleEstimator


def test_monocular_scale_is_explicit_and_stable():
    s = ScaleEstimator()
    assert s.estimate(None, None) == 1.0
    s.set_scale(2.5)
    assert s.estimate() == 2.5
    s.reset()
    assert s.scale == 1.0
