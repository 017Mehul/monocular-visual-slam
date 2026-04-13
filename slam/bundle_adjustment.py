# bundle_adjustment.py - Local Bundle Adjustment using scipy least squares
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from slam.pose_estimation import build_intrinsic_matrix


# ── Rodrigues helpers (avoid cv2 dependency in pure-math module) ──────────────

def _rodrigues_to_R(rvec: np.ndarray) -> np.ndarray:
	theta = np.linalg.norm(rvec)
	if theta < 1e-8:
		return np.eye(3)
	k = rvec / theta
	K = np.array([[0, -k[2], k[1]],
				  [k[2], 0, -k[0]],
				  [-k[1], k[0], 0]])
	return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _R_to_rodrigues(R: np.ndarray) -> np.ndarray:
	theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
	if theta < 1e-8:
		return np.zeros(3)
	k = np.array([R[2, 1] - R[1, 2],
				  R[0, 2] - R[2, 0],
				  R[1, 0] - R[0, 1]]) / (2 * np.sin(theta))
	return k * theta


def _project(K, rvec, tvec, pt3d):
	R = _rodrigues_to_R(rvec)
	p = R @ pt3d + tvec
	if p[2] < 1e-6:
		return np.array([0.0, 0.0])
	x = K[0, 0] * p[0] / p[2] + K[0, 2]
	y = K[1, 1] * p[1] / p[2] + K[1, 2]
	return np.array([x, y])


def _pack(camera_params, points_3d):
	return np.hstack([camera_params.ravel(), points_3d.ravel()])


def _unpack(x, n_cams, n_pts):
	cam_end = n_cams * 6
	camera_params = x[:cam_end].reshape(n_cams, 6)
	points_3d = x[cam_end:].reshape(n_pts, 3)
	return camera_params, points_3d


def _residuals(x, n_cams, n_pts, cam_indices, pt_indices, obs_2d, K):
	camera_params, points_3d = _unpack(x, n_cams, n_pts)
	res = []
	for cam_idx, pt_idx, obs in zip(cam_indices, pt_indices, obs_2d):
		rvec = camera_params[cam_idx, :3]
		tvec = camera_params[cam_idx, 3:]
		proj = _project(K, rvec, tvec, points_3d[pt_idx])
		res.extend(proj - obs)
	return np.array(res)


def _build_sparsity(n_cams, n_pts, cam_indices, pt_indices):
	n_obs = len(cam_indices)
	m = n_obs * 2
	n = n_cams * 6 + n_pts * 3
	A = lil_matrix((m, n), dtype=int)

	for i, (cam_idx, pt_idx) in enumerate(zip(cam_indices, pt_indices)):
		row = 2 * i
		A[row:row+2, cam_idx*6 : cam_idx*6+6] = 1
		col = n_cams * 6 + pt_idx * 3
		A[row:row+2, col:col+3] = 1

	return A


class BundleAdjuster:
	def __init__(self, max_iterations: int = 50, ftol: float = 1e-4):
		self.K = build_intrinsic_matrix()
		self.max_iter = max_iterations
		self.ftol = ftol

	def adjust(self, poses, points_3d, observations, fix_first=True):
		if len(poses) < 2 or len(points_3d) < 4 or len(observations) < 8:
			return poses, points_3d

		n_cams = len(poses)
		n_pts = len(points_3d)

		cam_params = np.zeros((n_cams, 6))
		for i, (R, t) in enumerate(poses):
			cam_params[i, :3] = _R_to_rodrigues(np.array(R))
			cam_params[i, 3:] = np.array(t).ravel()

		pts = np.array(points_3d, dtype=np.float64)

		cam_indices = np.array([o[0] for o in observations])
		pt_indices  = np.array([o[1] for o in observations])
		obs_2d      = np.array([[o[2], o[3]] for o in observations])

		x0 = _pack(cam_params, pts)
		sparsity = _build_sparsity(n_cams, n_pts, cam_indices, pt_indices)

		lower = np.full_like(x0, -np.inf)
		upper = np.full_like(x0,  np.inf)
		if fix_first:
			lower[:6] = x0[:6] - 1e-10
			upper[:6] = x0[:6] + 1e-10

		result = least_squares(
			_residuals,
			x0,
			jac_sparsity=sparsity,
			method="trf",
			args=(n_cams, n_pts, cam_indices, pt_indices, obs_2d, self.K),
			max_nfev=self.max_iter,
			ftol=self.ftol,
			verbose=0,
		)

		opt_cam, opt_pts = _unpack(result.x, n_cams, n_pts)

		opt_poses = []
		for i in range(n_cams):
			R = _rodrigues_to_R(opt_cam[i, :3])
			t = opt_cam[i, 3:].reshape(3, 1)
			opt_poses.append((R, t))

		cost_before = np.sum(_residuals(x0, n_cams, n_pts,
										cam_indices, pt_indices, obs_2d, self.K) ** 2)
		cost_after  = result.cost * 2
		print(f"[BA] cost {cost_before:.1f} → {cost_after:.1f} "
			  f"({n_cams} cams, {n_pts} pts, {len(observations)} obs)")

		return opt_poses, opt_pts