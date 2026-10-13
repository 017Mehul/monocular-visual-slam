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
            "--similarity-threshold", "0.85",
            "--revisit-radius-m", "0.5",
            "--min-separation-frames", "3",
            "--frame-stride", "1",
            "--vocabulary-training-frames", "2",
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


def test_loop_detector_checks_next_ranked_candidate_after_geometric_rejection(monkeypatch):
    from slam.loop_closure import Keyframe, LoopClosureDetector

    detector = LoopClosureDetector()
    descriptors = np.zeros((12, 32), dtype=np.uint8)
    kps = [cv2.KeyPoint(float(i), float(i), 8) for i in range(12)]
    candidates = [
        Keyframe(frame_idx=0, pose_idx=0, keypoints=kps, descriptors=descriptors),
        Keyframe(frame_idx=20, pose_idx=1, keypoints=kps, descriptors=descriptors),
        Keyframe(frame_idx=40, pose_idx=2, keypoints=kps, descriptors=descriptors),
    ]
    detector.keyframes = candidates
    detector.bow.query = lambda *args, **kwargs: [(0.95, candidates[0]), (0.90, candidates[1])]
    calls = []

    def verify(*args):
        candidate = args[-1]
        calls.append(candidate)
        if candidate is candidates[0]:
            return None, None, 8
        return np.eye(3), np.ones((3, 1)), detector.min_matches

    monkeypatch.setattr(detector, "_essential_verify", verify)
    found, selected, _, _ = detector.detect(frame_idx=100, keypoints=kps, descriptors=descriptors)
    assert found
    assert selected is candidates[1]
    assert calls == candidates[:2]



def test_loop_correction_requires_pose_prior_consistency():
    from slam.loop_closure import LoopClosureDetector

    candidate = np.eye(4)
    current = np.eye(4)
    current[0, 3] = 2.0

    assert LoopClosureDetector.pose_consistent(
        current, candidate, np.eye(3), np.array([[1.0], [0.0], [0.0]])
    )
    # A geometrically plausible match in the opposite direction is not a loop
    # constraint consistent with the current trajectory.
    assert not LoopClosureDetector.pose_consistent(
        current, candidate, np.eye(3), np.array([[-1.0], [0.0], [0.0]])
    )

    angle = np.deg2rad(45.0)
    bad_rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    assert not LoopClosureDetector.pose_consistent(
        current, candidate, bad_rotation, np.array([[1.0], [0.0], [0.0]])
    )
