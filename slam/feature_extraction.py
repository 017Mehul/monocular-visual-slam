import cv2
import numpy as np
from slam.config import ORB_PARAMS


class FeatureExtractor:
    def __init__(self):
        self.orb = cv2.ORB_create(
            nfeatures=ORB_PARAMS["nfeatures"],
            scaleFactor=ORB_PARAMS["scaleFactor"],
            nlevels=ORB_PARAMS["nlevels"],
            edgeThreshold=ORB_PARAMS["edgeThreshold"],
            fastThreshold=ORB_PARAMS["fastThreshold"],
            scoreType=ORB_PARAMS["scoreType"],
            patchSize=ORB_PARAMS["patchSize"],
            WTA_K=ORB_PARAMS["WTA_K"],
        )
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def extract(self, frame: np.ndarray):
        gray = self._to_gray(frame)
        gray = self.clahe.apply(gray)

        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        if keypoints:
            keypoints, descriptors = self._filter_weak(keypoints, descriptors)

        return keypoints, descriptors

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        if len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def _filter_weak(self, keypoints, descriptors, min_response: float = 5.0):
        if descriptors is None:
            return keypoints, descriptors
        pairs = [(kp, d) for kp, d in zip(keypoints, descriptors)
                 if kp.response >= min_response]
        if not pairs:
            return keypoints, descriptors
        kps, descs = zip(*pairs)
        return list(kps), np.array(descs)

    def draw_keypoints(self, frame: np.ndarray, keypoints) -> np.ndarray:
        return cv2.drawKeypoints(
            frame, keypoints, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
