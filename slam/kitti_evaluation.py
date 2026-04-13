"""Small KITTI evaluation utility: ATE (with Umeyama alignment) and RPE.

Usage:
  python kitti_evaluation.py --gt path/to/poses.txt --est path/to/trajectory.csv

The groundtruth file should be KITTI poses (each line: 12 numbers, 3x4 matrix).
The estimated trajectory may be the CSV produced by `Trajectory.save_positions_csv`
(`frame_idx,x,y,z`) or a plain Nx3 whitespace CSV of positions.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from typing import Tuple


def read_kitti_poses(path: Path) -> np.ndarray:
    """Read KITTI poses file (each line: 12 numbers -> 3x4 matrix).

    Returns: (N, 4, 4) homogeneous poses (camera-to-world).
    """
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 12:
        raise ValueError("KITTI poses must have 12 values per line (3x4)")
    poses = np.eye(4)[None, :, :].repeat(data.shape[0], axis=0)
    poses[:, :3, :4] = data.reshape(-1, 3, 4)
    return poses


def read_positions(path: Path) -> np.ndarray:
    """Read estimated positions.

    Supports CSV from `Trajectory.save_positions_csv` (frame_idx,x,y,z)
    or plain Nx3 whitespace/CSV files.
    Returns: (N, 3)
    """
    # Handle optional CSV header like 'frame_idx,x,y,z'
    with open(path, 'r', encoding='utf-8') as f:
        first = f.readline()
    # If first token isn't numeric, assume header and skip it
    def _is_number(s: str) -> bool:
        try:
            float(s)
            return True
        except Exception:
            return False

    if not _is_number(first.strip().split(',')[0]):
        data = np.loadtxt(path, delimiter=',', skiprows=1)
    else:
        data = np.loadtxt(path, delimiter=',')

    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] == 4:
        return data[:, 1:4]
    if data.shape[1] == 3:
        return data
    raise ValueError("Estimated trajectory file must be Nx3 or Nx4(frame_idx,x,y,z)")


def umeyama_alignment(A: np.ndarray, B: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute similarity (s,R,t) that best maps A -> B (A,B are Nx3 points).
    Returns scale s, rotation R (3x3), translation t (3,).
    """
    assert A.shape == B.shape
    n = A.shape[0]
    mu_A = A.mean(axis=0)
    mu_B = B.mean(axis=0)
    X = A - mu_A
    Y = B - mu_B
    cov = (Y.T @ X) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_A = (X ** 2).sum() / n
    # Handle degenerate case where A has zero variance (all points identical)
    if abs(var_A) < 1e-12:
        s = 1.0
        R = np.eye(3)
        t = mu_B - mu_A
        return s, R, t
    s = np.trace(np.diag(D) @ S) / var_A
    t = mu_B - s * R @ mu_A
    return s, R, t


def compute_ate(gt_positions: np.ndarray, est_positions: np.ndarray) -> float:
    """Compute RMSE ATE after similarity alignment (scale+rot+trans).
    Returns RMSE (meters).
    """
    if gt_positions.shape[0] != est_positions.shape[0]:
        m = min(gt_positions.shape[0], est_positions.shape[0])
        gt_positions = gt_positions[:m]
        est_positions = est_positions[:m]
    s, R, t = umeyama_alignment(est_positions, gt_positions)
    est_aligned = (s * (R @ est_positions.T)).T + t
    err = gt_positions - est_aligned
    rmse = np.sqrt((err ** 2).sum(axis=1).mean())
    return float(rmse)


def compute_rpe(gt_poses: np.ndarray, est_poses: np.ndarray, delta: int = 1) -> Tuple[float, float]:
    """Compute RPE (translation RMSE in meters, rotation RMSE in degrees)

    Both inputs are (N,4,4). Only overlapping frames are considered.
    """
    n = min(gt_poses.shape[0], est_poses.shape[0])
    gt = gt_poses[:n]
    est = est_poses[:n]
    trans_errs = []
    rot_errs = []
    for i in range(n - delta):
        # relative transforms
        T_gt = np.linalg.inv(gt[i]) @ gt[i + delta]
        T_est = np.linalg.inv(est[i]) @ est[i + delta]
        # error transform
        T_err = np.linalg.inv(T_gt) @ T_est
        t_err = T_err[:3, 3]
        trans_errs.append(np.linalg.norm(t_err))
        R_err = T_err[:3, :3]
        # rotation angle from trace
        angle = np.arccos(max(-1.0, min(1.0, (np.trace(R_err) - 1) / 2)))
        rot_errs.append(np.degrees(angle))
    return float(np.sqrt(np.mean(np.square(trans_errs)))), float(np.sqrt(np.mean(np.square(rot_errs))))


def poses_from_positions(positions: np.ndarray) -> np.ndarray:
    """Make homogeneous poses from positions assuming identity rotation.

    This is a lightweight way to compute RPE when only positions are available.
    Returns (N,4,4)
    """
    poses = np.eye(4)[None, :, :].repeat(positions.shape[0], axis=0)
    poses[:, :3, 3] = positions
    return poses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt", required=True, help="KITTI poses file (3x4 per line)")
    p.add_argument("--est", required=True, help="Estimated trajectory (CSV frame_idx,x,y,z or Nx3)")
    p.add_argument("--rpe-delta", type=int, default=1, help="Frame delta for RPE (default=1)")
    args = p.parse_args()

    gt_path = Path(args.gt)
    est_path = Path(args.est)

    gt_poses = read_kitti_poses(gt_path)
    gt_positions = gt_poses[:, :3, 3]
    est_positions = read_positions(est_path)

    ate = compute_ate(gt_positions, est_positions)

    # Prepare poses for RPE: use identity rotations for est if only positions present
    est_poses = poses_from_positions(est_positions)
    rpe_trans, rpe_rot = compute_rpe(gt_poses, est_poses, delta=args.rpe_delta)

    print(f"ATE (rmse) = {ate:.4f} m")
    print(f"RPE translation (rmse, delta={args.rpe_delta}) = {rpe_trans:.4f} m")
    print(f"RPE rotation (rmse, delta={args.rpe_delta}) = {rpe_rot:.4f} deg")


if __name__ == "__main__":
    main()
