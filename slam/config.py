"""config.py - Central configuration for the SLAM pipeline
"""
# ── Camera intrinsics ─────────────────────────────────────────────────────────
CAMERA_PARAMS = {
    "fx": 718.856,
    "fy": 718.856,
    "cx": 607.1928,
    "cy": 185.2157,
}

ORB_PARAMS = {
    "nfeatures": 3000,
    "scaleFactor": 1.2,
    "nlevels": 8,
    "edgeThreshold": 19,
    "fastThreshold": 10,
    "scoreType": 0,
    "patchSize": 31,
    "WTA_K": 2,
}

MATCHING_PARAMS = {
    "ratio_threshold": 0.72,
}

RANSAC_PARAMS = {
    "prob": 0.999,
    "threshold": 1.0,
    "min_inliers": 15,
}

TRIANGULATION_PARAMS = {
    "min_depth": 0.05,
    "max_depth": 200.0,
}

LOOP_CLOSURE_PARAMS = {
    "min_keyframe_interval": 15,
    "similarity_threshold": 0.80,
    "min_matches": 40,
    "min_skip_frames": 40,
}

PIPELINE_PARAMS = {
    "min_features": 200,
    "min_matches": 30,
    "min_inliers": 15,
    "frame_resize_scale": 0.5,
    "viz_every": 3,
    "prune_every": 60,
    "loop_check_every": 10,
    "max_map_points": 50000,
}
