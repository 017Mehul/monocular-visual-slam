from slam.sensor_scale import SensorScaleProvider

def test_sensor_scale_uses_robust_metric_ratio():
    sensor = SensorScaleProvider(window=3)
    assert sensor.update(0.5, 1.0) == 2.0
    assert sensor.update(0.4, 0.8) == 2.0
    assert sensor.update(0.25, 0.5) == 2.0
    assert sensor.scale == 2.0

def test_sensor_scale_ignores_invalid_motion():
    sensor = SensorScaleProvider()
    assert sensor.update(0.0, 1.0) is None
    assert sensor.update(1.0, -1.0) is None
