import sys
from pathlib import Path


repo_root = Path(__file__).resolve().parents[1]
# Add the slam module dir to path
sys.path.insert(0, str(repo_root / "slam"))

import numpy as np
import kitti_evaluation as ke


def make_gt_file(path: Path, n: int = 5):
    # write identity poses (3x4) repeated
    line = "1 0 0 0 0 1 0 0 0 0 1 0\n"
    path.write_text(line * n)


def make_est_positions(path: Path, n: int = 5):
    # CSV with frame_idx,x,y,z all zeros
    lines = ["frame_idx,x,y,z"]
    for i in range(n):
        lines.append(f"{i},0,0,0")
    path.write_text("\n".join(lines))


def test_kitti_eval_zero_error(tmp_path: Path):
    gt = tmp_path / "gt.txt"
    est = tmp_path / "est.csv"
    make_gt_file(gt, n=6)
    make_est_positions(est, n=6)

    gt_poses = ke.read_kitti_poses(gt)
    est_positions = ke.read_positions(est)

    ate = ke.compute_ate(gt_poses[:, :3, 3], est_positions)
    assert np.isclose(ate, 0.0, atol=1e-6)

    est_poses = ke.poses_from_positions(est_positions)
    rpe_trans, rpe_rot = ke.compute_rpe(gt_poses, est_poses, delta=1)
    assert np.isclose(rpe_trans, 0.0, atol=1e-6)
    assert np.isclose(rpe_rot, 0.0, atol=1e-6)
