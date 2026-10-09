"""Bag-of-Words place recognition for binary ORB descriptors.

Self-contained BoW backend for environments without DBoW2 bindings.
"""
from __future__ import annotations
from collections import defaultdict
import cv2
import numpy as np

class BinaryVocabulary:
    def __init__(self, words=256, iterations=20):
        self.words = int(words)
        self.iterations = int(iterations)
        self.centers = None

    def fit(self, descriptors):
        if descriptors is None or len(descriptors) < max(2, self.words):
            return
        data = np.asarray(descriptors, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.iterations, 0.5)
        _, _, centers = cv2.kmeans(data, self.words, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        self.centers = centers.astype(np.float32)

    def transform(self, descriptors):
        if self.centers is None or descriptors is None or len(descriptors) == 0:
            return np.empty((0,), dtype=np.int32)
        data = np.asarray(descriptors, dtype=np.float32)
        distances = ((data[:, None, :] - self.centers[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(distances, axis=1).astype(np.int32)

class BoWLoopClosure:
    """Incremental TF-IDF BoW place-recognition index.

    It proposes visually similar keyframes; existing Essential-matrix
    verification remains the geometric acceptance gate.
    """
    def __init__(self, words=256, min_interval=15, min_skip_frames=40):
        self.vocabulary = BinaryVocabulary(words)
        self.min_interval = min_interval
        self.min_skip_frames = min_skip_frames
        self.keyframes = []
        self.histograms = {}
        self.inverted = defaultdict(set)
        self._last_frame = -min_interval

    @staticmethod
    def _normalize(words, n_words):
        if len(words) == 0:
            return {}
        counts = np.bincount(words, minlength=n_words).astype(np.float64)
        counts /= max(float(counts.sum()), 1.0)
        return {int(i): float(v) for i, v in enumerate(counts) if v > 0}

    def _idf(self, word):
        n_docs = max(len(self.keyframes), 1)
        df = len(self.inverted.get(word, ()))
        return float(np.log((1.0 + n_docs) / (1.0 + df)) + 1.0)

    def _tfidf(self, hist):
        weighted = {w: tf * self._idf(w) for w, tf in hist.items()}
        norm = np.sqrt(sum(v * v for v in weighted.values()))
        return {w: v / norm for w, v in weighted.items()} if norm > 1e-12 else {}

    @staticmethod
    def similarity(a, b):
        if not a or not b:
            return 0.0
        if len(a) > len(b):
            a, b = b, a
        return float(sum(v * b.get(w, 0.0) for w, v in a.items()))

    def _reindex(self):
        self.inverted = defaultdict(set)
        for idx, hist in self.histograms.items():
            for word in hist:
                self.inverted[word].add(idx)
        for idx, hist in list(self.histograms.items()):
            self.histograms[idx] = self._tfidf(hist)

    def register(self, frame_idx, keyframe, descriptors):
        if descriptors is None or len(descriptors) < 10 or frame_idx - self._last_frame < self.min_interval:
            return
        if self.vocabulary.centers is None and len(descriptors) >= self.vocabulary.words:
            self.vocabulary.fit(np.asarray(descriptors, dtype=np.uint8))
        if self.vocabulary.centers is None:
            return
        words = self.vocabulary.transform(descriptors)
        hist = self._normalize(words, self.vocabulary.words)
        self.keyframes.append((frame_idx, keyframe, descriptors))
        self.histograms[len(self.keyframes) - 1] = hist
        self._reindex()
        self._last_frame = frame_idx

    def query(self, frame_idx, descriptors, top_k=5):
        if self.vocabulary.centers is None or descriptors is None or not self.keyframes:
            return []
        words = self.vocabulary.transform(descriptors)
        hist = self._tfidf(self._normalize(words, self.vocabulary.words))
        scored = []
        for idx, (kf_frame, kf, _) in enumerate(self.keyframes):
            if frame_idx - kf_frame <= self.min_skip_frames:
                continue
            scored.append((self.similarity(hist, self.histograms.get(idx, {})), kf))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]
