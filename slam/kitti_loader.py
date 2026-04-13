# kitti_loader.py - KITTI odometry dataset image sequence loader
import cv2
import numpy as np
import os
import glob


class KITTILoader:
    def __init__(self, sequence_path: str, camera: str = "image_0"):
        self.image_dir = os.path.join(sequence_path, camera)
        self.frames = sorted(glob.glob(os.path.join(self.image_dir, "*.png")))

        if not self.frames:
            self.frames = sorted(glob.glob(os.path.join(self.image_dir, "*.jpg")))

        self._idx = 0
        print(f"[KITTILoader] Sequence: {sequence_path}")
        print(f"[KITTILoader] Camera  : {camera}")
        print(f"[KITTILoader] Frames  : {len(self.frames)}")

    def isOpened(self) -> bool:
        return len(self.frames) > 0

    def read(self):
        if self._idx >= len(self.frames):
            return False, None
        frame = cv2.imread(self.frames[self._idx])
        self._idx += 1
        if frame is None:
            return False, None
        return True, frame

    def release(self):
        pass

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self.frames))
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            return float(self._idx)
        return 0.0

    def set(self, prop_id, value):
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            self._idx = int(value)
        return True

    def __len__(self):
        return len(self.frames)


def load_kitti_calib(sequence_path: str) -> dict:
    calib_path = os.path.join(sequence_path, "calib.txt")
    with open(calib_path) as f:
        for line in f:
            if line.startswith("P0:"):
                vals = list(map(float, line.strip().split()[1:]))
                return {
                    "fx": vals[0],
                    "fy": vals[5],
                    "cx": vals[2],
                    "cy": vals[6],
                }
    raise ValueError(f"P0 not found in {calib_path}")
