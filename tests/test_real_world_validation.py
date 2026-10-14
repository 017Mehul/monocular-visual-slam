import json
import sys

import scripts.real_world_validation as validation


def test_validation_runner_retains_failure_report_when_pipeline_raises(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "real_world_validation.py",
            "--source", "broken-video.avi",
            "--frames", "12",
            "--output-dir", str(tmp_path),
        ],
    )

    def fail(_args):
        raise RuntimeError("synthetic pipeline failure")

    monkeypatch.setattr(validation, "run", fail)

    assert validation.main() == 2
    report = json.loads((tmp_path / "validation_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAIL"
    assert report["frames_requested"] == 12
    assert report["frames_processed"] == 0
    assert "synthetic pipeline failure" in report["failures"][0]
    assert "FAIL" in capsys.readouterr().out


def test_validation_runner_supplies_all_pipeline_flags(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "real_world_validation.py",
            "--source", "sample.avi",
            "--frames", "2",
            "--output-dir", str(tmp_path),
        ],
    )
    captured = {}

    def fake_run(args):
        captured.update(vars(args))
        return {
            "runtime_validation": {
                "p95_frame_time_ms": 10.0,
                "tracking_success_rate": 1.0,
                "effective_fps": 30.0,
                "slow_frame_rate": 0.0,
            },
            "frames_processed": 2,
            "duration_sec": 0.1,
        }

    monkeypatch.setattr(validation, "run", fake_run)
    assert validation.main() == 0
    assert captured["enable_ba"] is False
    assert captured["enable_map_pnp"] is False
    report = json.loads((tmp_path / "validation_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
