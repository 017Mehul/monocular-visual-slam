# main.py - Monocular Visual SLAM full pipeline (package entry)

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from slam.bundle_adjustment import BundleAdjuster
from slam.feature_extraction import FeatureExtractor
from slam.feature_matching import FeatureMatcher
from slam.keyframe_manager import KeyframeManager
from slam.kitti_loader import KITTILoader, load_kitti_calib
from slam.loop_closure import LoopClosureDetector
from slam.map_manager import MapManager
from slam.pose_estimation import PoseEstimator
from slam.relocalization import Relocalizer
from slam.scale_estimator import ScaleEstimator
from slam.trajectory import Trajectory
from slam.triangulation import Triangulator
from slam.visualization import Visualizer
from slam.config import CAMERA_PARAMS, PIPELINE_PARAMS


LOGGER = logging.getLogger("slam")


def pick_video_file():
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return path if path else None
    except Exception as exc:
        LOGGER.warning("File picker unavailable: %s", exc)
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Monocular Visual SLAM")
    parser.add_argument("--source", type=str, default=None, help="Video file path, webcam index, or omit to open file picker")
    parser.add_argument("--scale", type=float, default=PIPELINE_PARAMS["frame_resize_scale"], help="Resize factor applied to each input frame")
    parser.add_argument("--width", type=int, default=None, help="Requested capture width")
    parser.add_argument("--height", type=int, default=None, help="Requested capture height")
    parser.add_argument("--no-viz", action="store_true", help="Disable 3D trajectory and map visualization")
    parser.add_argument("--headless", action="store_true", help="Disable all GUI windows and render loops")
    parser.add_argument("--no-ba", action="store_true", help="Disable bundle adjustment for lower latency")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after processing this many frames")
    parser.add_argument("--config-file", type=str, default=None, help="JSON file containing CAMERA_PARAMS and/or PIPELINE_PARAMS overrides")
    parser.add_argument("--output-dir", type=str, default="outputs/latest_run", help="Directory for logs, summaries, and trajectories")
    parser.add_argument("--save-trajectory", action="store_true", help="Write `trajectory_positions.csv` into the output directory")
    parser.add_argument("--summary-json", action="store_true", help="Write `run_summary.json` into the output directory")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Console and file log verbosity")
    parser.add_argument("--log-file", type=str, default=None, help="Optional explicit log file path. Defaults to <output-dir>/slam.log")
    return parser.parse_args()


def configure_logging(log_level: str, log_file: Path | None):
    level = getattr(logging, log_level.upper(), logging.INFO)
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=handlers, force=True)


def load_runtime_overrides(config_file: str | None):
    if not config_file:
        return
    path = Path(config_file)
    with path.open("r", encoding="utf-8") as handle:
        overrides = json.load(handle)
    camera_updates = overrides.get("CAMERA_PARAMS", {})
    pipeline_updates = overrides.get("PIPELINE_PARAMS", {})
    if camera_updates:
        CAMERA_PARAMS.update(camera_updates)
    if pipeline_updates:
        PIPELINE_PARAMS.update(pipeline_updates)
    LOGGER.info("Loaded runtime overrides from %s (camera=%d, pipeline=%d)", path, len(camera_updates), len(pipeline_updates))


def ensure_output_dir(path_str: str) -> Path:
    output_dir = Path(path_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def open_source(source, width=None, height=None):
    if os.path.isdir(source) and os.path.isdir(os.path.join(source, "image_0")):
        loader = KITTILoader(source, camera="image_0")
        if not loader.isOpened():
            raise RuntimeError(f"No images found in {source}/image_0")
        try:
            calib = load_kitti_calib(source)
            CAMERA_PARAMS.update(calib)
            LOGGER.info("KITTI calibration loaded from %s (fx=%.2f, cx=%.2f, cy=%.2f)", source, calib["fx"], calib["cx"], calib["cy"])
        except Exception as exc:
            LOGGER.warning("Could not load KITTI calib.txt: %s", exc)
        return loader
    try:
        idx = int(source)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open webcam {idx}")
        return cap
    except ValueError:
        pass
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input source: {source}")
    return cap


def preprocess(frame, scale):
    if scale != 1.0:
        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(frame, (3, 3), 0)


def scale_intrinsics(scale):
    if scale == 1.0:
        return
    for key in ("fx", "fy", "cx", "cy"):
        CAMERA_PARAMS[key] *= scale


def show_frame(frame, viz, stats, enable_display):
    if not enable_display:
        return
    img = viz.draw_hud(frame, stats) if viz else frame
    cv2.imshow("SLAM (q=quit)", img)
    cv2.waitKey(1)


def should_quit(enable_display):
    return enable_display and (cv2.waitKey(1) & 0xFF == ord("q"))


def write_summary(output_dir: Path, summary: dict):
    path = output_dir / "run_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def run(args):
    output_dir = ensure_output_dir(args.output_dir)
    log_file = Path(args.log_file) if args.log_file else output_dir / "slam.log"
    configure_logging(args.log_level, log_file)
    load_runtime_overrides(args.config_file)

    if args.source is None:
        picked = pick_video_file()
        if picked:
            args.source = picked
            LOGGER.info("Selected input source via picker: %s", picked)
        else:
            args.source = "0"
            LOGGER.info("No file selected, defaulting to webcam 0")

    scale_intrinsics(args.scale)

    cap = None
    viz = None
    summary = {"source": args.source, "scale": args.scale, "headless": args.headless, "bundle_adjustment": not args.no_ba, "output_dir": str(output_dir), "start_time_epoch": time.time(), "frames_processed": 0, "successful_tracking_frames": 0, "skipped_feature_frames": 0, "skipped_match_frames": 0, "skipped_inlier_frames": 0, "relocalizations": 0, "loop_closures": 0, "keyframes": 0, "final_map_points": 0, "total_map_points_added": 0, "avg_fps_estimate": 0.0}

    try:
        cap = open_source(args.source, args.width, args.height)

        extractor = FeatureExtractor()
        matcher = FeatureMatcher()
        estimator = PoseEstimator()
        tri = Triangulator()
        map_mgr = MapManager()
        loop_det = LoopClosureDetector()
        trajectory = Trajectory()
        kf_mgr = KeyframeManager()
        scale_est = ScaleEstimator()
        reloc = Relocalizer()
        ba = BundleAdjuster() if not args.no_ba else None
        if not args.no_viz and not args.headless:
            viz = Visualizer()

        cfg = PIPELINE_PARAMS
        min_feat = cfg["min_features"]
        min_match = cfg["min_matches"]
        min_inliers = cfg["min_inliers"]
        viz_every = cfg["viz_every"]
        prune_every = cfg["prune_every"]
        loop_every = cfg["loop_check_every"]
        ba_every = 20
        display_enabled = not args.headless

        lc_frames = []
        prev_pts3d = None
        frame_idx = 0
        fps = 0.0
        kf_since_ba = 0

        prev_frame = prev_kp = prev_desc = None
        LOGGER.info("Waiting for first usable frame")
        while prev_frame is None:
            ret, raw = cap.read()
            if not ret:
                raise RuntimeError("Cannot read from source during bootstrap")
            frame = preprocess(raw, args.scale)
            kp, desc = extractor.extract(frame)
            if desc is not None and len(kp) >= min_feat:
                prev_frame, prev_kp, prev_desc = frame, kp, desc
                LOGGER.info("Bootstrap complete with %d features", len(kp))

        LOGGER.info("SLAM loop started")

        while True:
            if args.max_frames is not None and frame_idx >= args.max_frames:
                LOGGER.info("Reached frame limit (%d); stopping", args.max_frames)
                break

            ret, raw = cap.read()
            if not ret:
                LOGGER.info("Input stream ended")
                break

            t0 = time.time()
            frame = preprocess(raw, args.scale)
            frame_idx += 1
            summary["frames_processed"] = frame_idx

            curr_kp, curr_desc = extractor.extract(frame)
            n_feat = len(curr_kp) if curr_kp else 0
            stats = {"frame_idx": frame_idx, "fps": fps, "features": n_feat, "matches": 0, "inliers": 0, "map_pts": map_mgr.size(), "ba_window": 0}

            if curr_desc is None or n_feat < min_feat:
                summary["skipped_feature_frames"] += 1
                reloc.update_state(False)
                LOGGER.warning("[%04d] Skipping frame: features=%d", frame_idx, n_feat)
                show_frame(frame, viz, stats, display_enabled)
                if should_quit(display_enabled):
                    LOGGER.info("User requested exit")
                    break
                continue

            if reloc.is_lost():
                r_rel, t_rel, ok = reloc.relocalize(curr_kp, curr_desc)
                if ok:
                    trajectory.update(r_rel, t_rel)
                    prev_frame, prev_kp, prev_desc = frame, curr_kp, curr_desc
                    summary["relocalizations"] += 1
                    LOGGER.info("[%04d] Relocalization succeeded", frame_idx)
                else:
                    LOGGER.warning("[%04d] Relocalization still pending", frame_idx)
                    show_frame(frame, viz, stats, display_enabled)
                    if should_quit(display_enabled):
                        LOGGER.info("User requested exit")
                        break
                    continue

            matches = matcher.match(prev_desc, curr_desc)
            n_match = len(matches)
            stats["matches"] = n_match

            if n_match < min_match:
                summary["skipped_match_frames"] += 1
                reloc.update_state(False)
                LOGGER.warning("[%04d] Skipping frame: matches=%d", frame_idx, n_match)
                show_frame(frame, viz, stats, display_enabled)
                if should_quit(display_enabled):
                    LOGGER.info("User requested exit")
                    break
                continue

            pts_prev, pts_curr = matcher.get_matched_points(prev_kp, curr_kp, matches)

            r_mat, t_vec, inlier_mask, ok = estimator.estimate(pts_prev, pts_curr)
            n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
            stats["inliers"] = n_inliers

            tracking_ok = ok and n_inliers >= min_inliers
            reloc.update_state(tracking_ok)
            if not tracking_ok:
                summary["skipped_inlier_frames"] += 1
                LOGGER.warning("[%04d] Skipping frame: inliers=%d", frame_idx, n_inliers)
                show_frame(frame, viz, stats, display_enabled)
                if should_quit(display_enabled):
                    LOGGER.info("User requested exit")
                    break
                continue

            summary["successful_tracking_frames"] += 1

            pts_prev_in, pts_curr_in = estimator.filter_inliers(pts_prev, pts_curr, inlier_mask)

            scale = scale_est.estimate(prev_pts3d, None, r_mat, t_vec)
            t_scaled = t_vec * scale

            trajectory.update(r_mat, t_scaled)
            r_world, t_world = trajectory.get_latest_Rt()

            new_pts = tri.triangulate(r_world, t_world, np.eye(3), np.zeros((3, 1)), pts_prev_in, pts_curr_in)
            map_mgr.add_points(new_pts)
            prev_pts3d = new_pts if len(new_pts) > 0 else prev_pts3d
            stats["map_pts"] = map_mgr.size()

            if kf_mgr.should_insert(frame_idx, r_mat, t_scaled, n_match, len(prev_kp)):
                kf_mgr.insert(frame_idx, r_world, t_world, curr_kp, curr_desc, new_pts)
                reloc.add_frame(frame_idx, curr_kp, curr_desc, new_pts, r_world, t_world)
                kf_since_ba += 1

                if ba and kf_since_ba >= ba_every and kf_mgr.size() >= 3:
                    observations, win_pts = kf_mgr.build_observations()
                    if len(observations) >= 8 and len(win_pts) >= 4:
                        opt_poses, _ = ba.adjust(kf_mgr.get_window_poses(), win_pts, observations)
                        kf_mgr.update_poses_from_ba(opt_poses)
                    kf_since_ba = 0

            loop_det.register_keyframe(frame_idx, trajectory.length() - 1, curr_kp, curr_desc)
            if frame_idx % loop_every == 0:
                found, kf, _, _ = loop_det.detect(frame_idx, curr_kp, curr_desc)
                if found and kf is not None:
                    lc_frames.append(trajectory.length() - 1)
                    correction = loop_det.compute_correction(trajectory.get_latest_pose(), trajectory.poses[kf.pose_idx])
                    loop_det.apply_smooth_correction(trajectory.poses, kf.pose_idx + 1, correction)
                    summary["loop_closures"] += 1
                    LOGGER.info("[%04d] Loop closure applied", frame_idx)

            if frame_idx % prune_every == 0:
                map_mgr.prune_outliers()

            fps = 0.9 * fps + 0.1 / max(time.time() - t0, 1e-6)
            stats["fps"] = fps
            stats["ba_window"] = len(kf_mgr.get_window_poses()) if kf_mgr.size() > 0 else 0
            summary["avg_fps_estimate"] = fps
            summary["keyframes"] = kf_mgr.size()
            summary["final_map_points"] = map_mgr.size()
            summary["total_map_points_added"] = map_mgr.total_added

            if display_enabled:
                match_img = matcher.draw_matches(prev_frame, prev_kp, frame, curr_kp, matches)
                if viz:
                    match_img = viz.draw_hud(match_img, stats)
                cv2.imshow("SLAM (q=quit)", match_img)

            if viz and frame_idx % viz_every == 0:
                viz.update(trajectory.get_positions(), map_mgr.get_point_cloud(), lc_frames, kf_mgr.get_window_poses())

            LOGGER.info("[%04d] fps=%.1f feat=%d match=%d inlier=%d scale=%.3f new3d=%d map=%d kfs=%d state=%s", frame_idx, fps, n_feat, n_match, n_inliers, scale, len(new_pts), map_mgr.size(), kf_mgr.size(), reloc.state.value)

            prev_frame, prev_kp, prev_desc = frame, curr_kp, curr_desc

            if should_quit(display_enabled):
                LOGGER.info("User requested exit")
                break

        summary["end_time_epoch"] = time.time()
        summary["duration_sec"] = round(summary["end_time_epoch"] - summary["start_time_epoch"], 3)

        if args.save_trajectory:
            trajectory.save_positions_csv(output_dir / "trajectory_positions.csv")
            LOGGER.info("Saved trajectory to %s", output_dir / "trajectory_positions.csv")

        if args.summary_json:
            summary_path = write_summary(output_dir, summary)
            LOGGER.info("Saved run summary to %s", summary_path)

        LOGGER.info("Run complete: frames=%d map=%d total_added=%d relocalizations=%d loops=%d", summary["frames_processed"], summary["final_map_points"], summary["total_map_points_added"], summary["relocalizations"], summary["loop_closures"])
        return summary
    finally:
        if cap is not None:
            cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        if viz is not None:
            viz.close()


if __name__ == "__main__":
    run(parse_args())
