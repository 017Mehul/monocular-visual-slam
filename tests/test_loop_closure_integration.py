import json
import sys

import cv2
import numpy as np

import scripts.evaluate_loop_closure as evaluator


def test_loop_evaluation_recovers_known_synthetic_revisits(tmp_path, monkeypatch):
    sequence = tmp_path / "sequence_00"
    images = sequence / "image_0"
    images.mkdir(parents=True)
    (sequence / "calib.txt").write_text("P0: synthetic calibration placeholder\n", encoding="utf-8")

    rng = np.random.default_rng(2026)
    views = [
        rng.integers(0, 256, size=(240, 320), dtype=np.uint8),
        rng.integers(0, 256, size=(240, 320), dtype=np.uint8),
        rng.integers(0, 256, size=(240, 320), dtype=np.uint8),
        rng.integers(0, 256, size=(240, 320), dtype=np.uint8),
    ]
    frames = [views[0], views[1], views[2], views[3], views[0], views[1]]
    for frame_idx, image in enumerate(frames):
        assert cv2.imwrite(str(images / f"{frame_idx:06d}.png"), image)

    x_positions = [0.0, 10.0, 20.0, 30.0, 0.0, 10.0]
    gt = tmp_path / "poses.txt"
    gt.write_text(
        "\n".join(
            " ".join(map(str, [1, 0, 0, x, 0, 1, 0, 0, 0, 0, 1, 0]))
            for x in x_positions
        ) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "loop_metrics.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_loop_closure.py",
            "--sequence", str(sequence),
            "--ground-truth", str(gt),
            "--output", str(output),
            "--words", "32",
            "--similarity-threshold", "0.1",
            "--revisit-radius-m", "0.5",
            "--min-separation-frames", "3",
            "--frame-stride", "1",
        ],
    )

    assert evaluator.main() == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["positive_revisit_queries"] == 2
    assert result["true_positives"] == 2
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
