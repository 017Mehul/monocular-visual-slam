#!/usr/bin/env python3
"""Plot KITTI ground-truth and estimated trajectories and save PNG.

Usage:
  python tools/plot_kitti_trajectory.py --gt dataset/dataset/poses/00.txt \
      --est outputs/kitti_00/trajectory_positions.csv --out docs/kitti_trajectory.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_kitti_poses(path: Path) -> np.ndarray:
    data = np.loadtxt(path, dtype=float)
    poses = data.reshape(-1, 3, 4)
    positions = poses[:, :3, 3]
    return positions


def read_est_positions(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    # expected frame_idx,x,y,z -> return frames and positions
    if data.ndim == 1:
        data = data.reshape(1, -1)
    frames = data[:, 0].astype(int)
    positions = data[:, 1:4]
    return frames, positions


def umeyama_alignment(A: np.ndarray, B: np.ndarray):
    # A, B: Nx3
    mu_A = A.mean(axis=0)
    mu_B = B.mean(axis=0)
    AA = A - mu_A
    BB = B - mu_B
    cov = AA.T @ BB / A.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var_A = (AA ** 2).sum() / A.shape[0]
    if var_A == 0:
        s = 1.0
    else:
        s = np.trace(np.diag(D) @ S) / var_A
    t = mu_B - s * R @ mu_A
    return s, R, t


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--est", required=True)
    ap.add_argument("--out", default="docs/kitti_trajectory.png")
    args = ap.parse_args(argv)

    gt = read_kitti_poses(Path(args.gt))
    frames, est = read_est_positions(Path(args.est))

    # pick GT positions corresponding to estimated frames
    if frames.max() >= gt.shape[0]:
        raise SystemExit(f"Estimated frame index {frames.max()} exceeds GT length {gt.shape[0]}")

    gt_sel = gt[frames]

    # align est -> gt_sel
    s, R, t = umeyama_alignment(est, gt_sel)
    est_aligned = (s * (R @ est.T)).T + t

    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(gt_sel[:, 0], gt_sel[:, 2], "-", color="#1f77b4", label="GT (selected frames)")
    plt.plot(est_aligned[:, 0], est_aligned[:, 2], "--", color="#ff7f0e", label="Estimate (aligned)")
    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title("KITTI Sequence 00: Ground Truth vs Estimated Trajectory")
    plt.legend()
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outp, bbox_inches="tight")
    print("Saved trajectory plot to", outp)


if __name__ == "__main__":
    raise SystemExit(main())
