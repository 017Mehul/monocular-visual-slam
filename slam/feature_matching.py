import cv2
import numpy as np
from slam.config import MATCHING_PARAMS


class FeatureMatcher:
    def __init__(self):
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.ratio = MATCHING_PARAMS["ratio_threshold"]

    def match(self, desc1: np.ndarray, desc2: np.ndarray):
        if desc1 is None or desc2 is None:
            return []
        if len(desc1) < 2 or len(desc2) < 2:
            return []

        desc1 = np.array(desc1, dtype=np.uint8)
        desc2 = np.array(desc2, dtype=np.uint8)

        raw = self.matcher.knnMatch(desc1, desc2, k=2)

        good = []
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio * n.distance:
                good.append(m)

        good = self._deduplicate(good)
        return good

    def _deduplicate(self, matches):
        best = {}
        for m in matches:
            tid = m.trainIdx
            if tid not in best or m.distance < best[tid].distance:
                best[tid] = m
        return list(best.values())

    def get_matched_points(self, kp1, kp2, matches):
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        return pts1, pts2

    def draw_matches(self, img1, kp1, img2, kp2, matches, max_draw=60) -> np.ndarray:
        return cv2.drawMatches(
            img1, kp1, img2, kp2, matches[:max_draw], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
