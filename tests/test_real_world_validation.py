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
