"""Simple IMU interface scaffold for future sensor-fusion integration.

This module provides a minimal `IMU` class that can load timestamped
accelerometer/gyroscope CSV logs and stream samples to the SLAM pipeline.

The implementation here is intentionally lightweight: it does not perform
calibration, bias estimation, or sophisticated preintegration — it's a
convenient place to add those features later and to keep IMU I/O isolated
from the rest of the codebase.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np


@dataclass
class IMUSample:
    timestamp: float
    accel: np.ndarray  # shape (3,)
    gyro: np.ndarray   # shape (3,)


class IMU:
    """Minimal IMU loader / iterator.

    Expected CSV format (header optional): timestamp,ax,ay,az,gx,gy,gz
    Timestamps in seconds (float). Accelerations in m/s^2, gyro in rad/s.
    """

    def __init__(self, csv_path: Optional[str] = None):
        self.samples: list[IMUSample] = []
        if csv_path:
            self.load_csv(Path(csv_path))

    def load_csv(self, path: Path) -> None:
        data = np.loadtxt(path, delimiter=',')
        if data.ndim == 1:
            data = data.reshape(1, -1)
        for row in data:
            t = float(row[0])
            ax, ay, az, gx, gy, gz = map(float, row[1:7])
            self.samples.append(IMUSample(t, np.array([ax, ay, az]), np.array([gx, gy, gz])))

    def __iter__(self) -> Iterator[IMUSample]:
        return iter(self.samples)

    def get_samples_between(self, t0: float, t1: float) -> list[IMUSample]:
        return [s for s in self.samples if t0 <= s.timestamp <= t1]

    def is_available(self) -> bool:
        return len(self.samples) > 0
