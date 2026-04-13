# visualization.py - Real-time trajectory and point cloud visualization

import time
import numpy as np
import cv2

try:
    import open3d as o3d
    _OPEN3D = True
except ImportError:
    _OPEN3D = False
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

VIZ_MAX_DEPTH  = 80.0
VIZ_MAX_POINTS = 20_000
VIZ_BA_WINDOW  = 7


class Visualizer:
    def __init__(self):
        self._last_update = time.time()
        if _OPEN3D:
            self._init_open3d()
        else:
            print("[Visualizer] Open3D not found — using matplotlib fallback.")
            self._init_matplotlib()

    def _init_open3d(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Monocular Visual SLAM", width=1280, height=720)
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)
        self.traj_ls = o3d.geometry.LineSet()
        self.vis.add_geometry(self.traj_ls)
        self.ba_ls = o3d.geometry.LineSet()
        self.vis.add_geometry(self.ba_ls)
        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.05, 0.05, 0.05])
        opt.point_size = 1.5
        self._first_update = True

    def _init_matplotlib(self):
        plt.ion()
        self.fig = plt.figure(figsize=(13, 5))
        self.ax2d = self.fig.add_subplot(121)
        self.ax3d = self.fig.add_subplot(122, projection="3d")
        self.fig.suptitle("Monocular Visual SLAM", fontsize=13)

    def update(self, traj: np.ndarray, cloud: np.ndarray,
               lc_indices: list = None, ba_window_poses: list = None):
        if _OPEN3D:
            self._update_open3d(traj, cloud, lc_indices, ba_window_poses)
        else:
            self._update_matplotlib(traj, cloud, lc_indices, ba_window_poses)

    def _filter_cloud(self, cloud: np.ndarray) -> np.ndarray:
        if len(cloud) == 0:
            return cloud
        mask = cloud[:, 2] > 0
        mask &= np.linalg.norm(cloud, axis=1) < VIZ_MAX_DEPTH
        return cloud[mask]

    def _update_open3d(self, traj, cloud, lc_indices, ba_window_poses):
        filtered = self._filter_cloud(np.array(cloud) if len(cloud) else np.empty((0, 3)))
        if len(filtered) > VIZ_MAX_POINTS:
            filtered = filtered[-VIZ_MAX_POINTS:]
        self.pcd.points = o3d.utility.Vector3dVector(filtered)
        if len(filtered) > 0:
            y = filtered[:, 1]
            y_norm = (y - y.min()) / (y.ptp() + 1e-8)
            colors = _height_colormap(y_norm)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
        self.vis.update_geometry(self.pcd)
        if len(traj) > 1:
            pts = np.array(traj, dtype=np.float64)
            lines = [[i, i + 1] for i in range(len(pts) - 1)]
            colors_traj = [[0.2, 0.6, 1.0]] * len(lines)
            if lc_indices:
                for li in lc_indices:
                    if 0 < li < len(lines):
                        colors_traj[li - 1] = [1.0, 0.3, 0.1]
            self.traj_ls.points = o3d.utility.Vector3dVector(pts)
            self.traj_ls.lines  = o3d.utility.Vector2iVector(lines)
            self.traj_ls.colors = o3d.utility.Vector3dVector(colors_traj)
            self.vis.update_geometry(self.traj_ls)
        if ba_window_poses and len(ba_window_poses) > 1:
            ba_pts = np.array([t.flatten() for _, t in ba_window_poses], dtype=np.float64)
            ba_lines = [[i, i + 1] for i in range(len(ba_pts) - 1)]
            ba_colors = [[1.0, 0.9, 0.0]] * len(ba_lines)
            self.ba_ls.points = o3d.utility.Vector3dVector(ba_pts)
            self.ba_ls.lines  = o3d.utility.Vector2iVector(ba_lines)
            self.ba_ls.colors = o3d.utility.Vector3dVector(ba_colors)
            self.vis.update_geometry(self.ba_ls)
        if self._first_update and len(traj) > 1:
            self.vis.reset_view_point(True)
            self._first_update = False
        self.vis.poll_events()
        self.vis.update_renderer()

    def _update_matplotlib(self, traj, cloud, lc_indices, ba_window_poses):
        self.ax2d.cla()
        self.ax3d.cla()
        self.ax2d.set_title("Trajectory (X-Z top-down)")
        self.ax2d.set_xlabel("X")
        self.ax2d.set_ylabel("Z")
        self.ax2d.grid(True)
        if len(traj) > 1:
            t = np.array(traj)
            self.ax2d.plot(t[:, 0], t[:, 2], "b-", lw=1.5)
            self.ax2d.scatter(*t[0, [0, 2]],  c="green", s=80, zorder=5, label="Start")
            self.ax2d.scatter(*t[-1, [0, 2]], c="red",   s=80, zorder=5, label="Now")
            if lc_indices:
                valid = [i for i in lc_indices if i < len(t)]
                if valid:
                    self.ax2d.scatter(t[valid, 0], t[valid, 2],
                                      c="orange", s=120, marker="*", zorder=6, label="Loop")
            if ba_window_poses and len(ba_window_poses) > 1:
                ba_pts = np.array([tv.flatten() for _, tv in ba_window_poses])
                self.ax2d.plot(ba_pts[:, 0], ba_pts[:, 2],
                               "y-", lw=3, alpha=0.7, label="BA window")
            self.ax2d.legend(fontsize=8)
        self.ax3d.set_title("Sparse 3D Map")
        self.ax3d.set_xlabel("X")
        self.ax3d.set_ylabel("Y")
        self.ax3d.set_zlabel("Z")
        if len(cloud) > 0:
            c = self._filter_cloud(np.array(cloud))
            if len(c) > 0:
                if len(c) > VIZ_MAX_POINTS:
                    c = c[-VIZ_MAX_POINTS:]
                step = max(1, len(c) // 3000)
                pc = c[::step]
                self.ax3d.scatter(pc[:, 0], pc[:, 1], pc[:, 2],
                                  s=0.4, c="cyan", alpha=0.4)
        if len(traj) > 1:
            t = np.array(traj)
            self.ax3d.plot(t[:, 0], t[:, 1], t[:, 2], "r-", lw=1.2, label="Trajectory")
            self.ax3d.legend(fontsize=8)
        plt.tight_layout()
        plt.pause(0.001)

    def draw_hud(self, frame: np.ndarray, stats: dict) -> np.ndarray:
        out = frame.copy()
        lines = [
            f"Frame : {stats.get('frame_idx', 0):04d}",
            f"FPS   : {stats.get('fps', 0):.1f}",
            f"Feat  : {stats.get('features', 0)}",
            f"Match : {stats.get('matches', 0)}",
            f"Inlier: {stats.get('inliers', 0)}",
            f"MapPts: {stats.get('map_pts', 0)}",
            f"BA win: {stats.get('ba_window', 0)} kfs",
        ]
        y = 20
        for line in lines:
            cv2.putText(out, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
            y += 22
        return out

    def close(self):
        if _OPEN3D:
            self.vis.destroy_window()
        else:
            plt.ioff()
            plt.show()


def _height_colormap(values: np.ndarray) -> np.ndarray:
    r = values
    g = 1.0 - values * 0.5
    b = 1.0 - values * 0.3
    return np.stack([r, g, b], axis=1).clip(0, 1)
