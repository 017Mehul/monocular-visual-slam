import pytest

from slam.sensor_scale import SensorScaleProvider, load_metric_scale_csv


def test_sensor_scale_recovers_known_synthetic_motion():
    sensor = SensorScaleProvider(window=5)
    # Metric displacements are exactly 3x the visual translation magnitudes.
    for visual, metric in [(0.10, 0.30), (0.20, 0.60), (0.15, 0.45), (0.30, 0.90)]:
        assert sensor.update(visual, metric) == pytest.approx(3.0)
    assert sensor.scale == pytest.approx(3.0)


def test_invalid_motion_does_not_create_or_overwrite_scale():
    sensor = SensorScaleProvider()
    assert sensor.update(0.0, 1.0) is None
    assert sensor.update(1.0, -1.0) is None
    assert sensor.scale is None
    assert sensor.update(0.5, 1.0) == pytest.approx(2.0)
    assert sensor.update(float("nan"), 5.0) is None
    assert sensor.scale == pytest.approx(2.0)


def test_csv_contract_accepts_header_and_consecutive_per_frame_displacements(tmp_path):
    path = tmp_path / "motion.csv"
    path.write_text("frame_idx,metric_delta_m\n1,0.3\n2,0.6\n3,0.45\n", encoding="utf-8")
    assert load_metric_scale_csv(path) == {1: 0.3, 2: 0.6, 3: 0.45}


@pytest.mark.parametrize("body, message", [
    ("frame_idx,metric_delta_m\n1,0.2\n3,0.4\n", "Non-contiguous"),
    ("frame_idx,metric_delta_m\n1,-0.2\n", "non-negative"),
    ("frame_idx,metric_delta_m\n1,0.2\n1,0.3\n", "Duplicate"),
    ("frame_idx,metric_translation_m\n1,0.2\n", "header"),
])
def test_csv_rejects_invalid_or_unsynchronized_samples(tmp_path, body, message):
    path = tmp_path / "bad.csv"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_metric_scale_csv(path)
