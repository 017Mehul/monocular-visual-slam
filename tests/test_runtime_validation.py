from slam.runtime_validation import RuntimeMonitor


def test_runtime_monitor_tracks_health_metrics():
    monitor = RuntimeMonitor(window_size=10, max_frame_time_ms=100.0)
    monitor.record(0.02, 300, 100, 50, True)
    monitor.record(0.20, 100, 20, 5, False)
    report = monitor.snapshot()
    assert report["frames_observed"] == 2
    assert report["tracking_success_rate"] == 0.5
    assert report["slow_frame_rate"] == 0.5
    assert report["avg_features"] == 200.0
    assert report["p95_frame_time_ms"] >= 200.0
    assert report["p99_frame_time_ms"] >= 200.0
