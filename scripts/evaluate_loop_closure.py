#!/usr/bin/env python3
"""Measure BoW loop-candidate precision/recall against KITTI ground-truth revisits."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from slam.bow_loop_closure import BoWLoopClosure
from slam.config import LOOP_CLOSURE_PARAMS
from slam.kitti_evaluation import read_kitti_poses


def precision_recall(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else None)
    return {"true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, help="KITTI sequence directory containing image_0/ and calib.txt")
    parser.add_argument("--ground-truth", required=True, help="KITTI 3x4 ground-truth poses file")
    parser.add_argument("--output", default="outputs/loop_closure_pr.json")
    parser.add_argument("--revisit-radius-m", type=float, default=5.0)
    parser.add_argument("--min-separation-frames", type=int, default=100)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--words", type=int, default=LOOP_CLOSURE_PARAMS.get("bow_words", 256))
    parser.add_argument("--similarity-threshold", type=float, default=LOOP_CLOSURE_PARAMS["similarity_threshold"])
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    if args.revisit_radius_m <= 0 or args.min_separation_frames < 1 or args.frame_stride < 1 or args.words < 2:
        parser.error("radius and words must be positive; frame separation and stride must be >= 1")

    sequence = Path(args.sequence)
    image_dir = sequence / "image_0"
    if not image_dir.is_dir() or not (sequence / "calib.txt").is_file():
        parser.error("sequence must contain image_0/ and sequence-specific calib.txt")
    image_paths = sorted(image_dir.glob("*.png")) or sorted(image_dir.glob("*.jpg"))
    gt = read_kitti_poses(Path(args.ground_truth))
    count = min(len(image_paths), len(gt))
    if args.max_frames is not None:
        count = min(count, args.max_frames)
    if count < 2:
        parser.error("sequence and ground truth must provide at least two aligned frames")

    orb = cv2.ORB_create(nfeatures=3000)
    bow = BoWLoopClosure(words=args.words, min_interval=args.frame_stride,
                         min_skip_frames=args.min_separation_frames)
    tp = fp = fn = queries = 0
    for frame_idx in range(0, count, args.frame_stride):
        image = cv2.imread(str(image_paths[frame_idx]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        _, descriptors = orb.detectAndCompute(image, None)
        if descriptors is None or len(descriptors) < 10:
            continue

        candidate = None
        if bow.vocabulary.centers is not None:
            ranked = bow.query(frame_idx, descriptors, top_k=1)
            if ranked and ranked[0][0] >= args.similarity_threshold:
                candidate = ranked[0][1]

        known_revisits = [
            previous_idx
            for previous_idx, _, _ in bow.keyframes
            if frame_idx - previous_idx >= args.min_separation_frames
            and np.linalg.norm(gt[frame_idx, :3, 3] - gt[previous_idx, :3, 3]) <= args.revisit_radius_m
        ]
        if known_revisits:
            queries += 1
            if candidate is not None:
                if candidate["frame_idx"] in known_revisits:
                    tp += 1
                else:
                    fp += 1
            else:
                fn += 1
        elif candidate is not None:
            fp += 1

        bow.register(frame_idx, {"frame_idx": frame_idx}, descriptors)

    result = {
        "sequence": sequence.name,
        "frames_considered": count,
        "registered_keyframes": len(bow.keyframes),
        "revisit_radius_m": args.revisit_radius_m,
        "min_separation_frames": args.min_separation_frames,
        "frame_stride": args.frame_stride,
        "similarity_threshold": args.similarity_threshold,
        "positive_revisit_queries": queries,
        **precision_recall(tp, fp, fn),
        "protocol_note": "Ground-truth position proximity labels revisits; this is a candidate-retrieval diagnostic, not a full geometric loop-closure evaluation.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if bow.keyframes and queries else 2


if __name__ == "__main__":
    raise SystemExit(main())
