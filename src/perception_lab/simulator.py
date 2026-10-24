"""Simulated municipal traffic-camera frames for local demos and tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perception_lab.detector import CLASSES, Detection


@dataclass(frozen=True)
class SimulatedFrame:
    frame: np.ndarray
    ground_truth: list[Detection]
    camera_id: str
    corridor_id: str
    frame_index: int


class FrameSimulator:
    """
    Generates synthetic RGB frames representing urban corridor traffic.

    Useful for CI and demos without live camera infrastructure.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 360,
        seed: int = 42,
        camera_id: str = "cam-avenue-hassan-01",
        corridor_id: str = "corridor-tetouan-centre",
    ) -> None:
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)
        self.camera_id = camera_id
        self.corridor_id = corridor_id
        self._index = 0

    def next_frame(
        self,
        *,
        n_vehicles: int | None = None,
        n_pedestrians: int | None = None,
        congestion_level: float = 0.4,
    ) -> SimulatedFrame:
        """Paint a procedural road scene and return frame + ground-truth boxes."""
        congestion_level = float(np.clip(congestion_level, 0.0, 1.0))
        if n_vehicles is None:
            n_vehicles = int(self.rng.integers(2, 8 + int(12 * congestion_level)))
        if n_pedestrians is None:
            n_pedestrians = int(self.rng.integers(0, 3 + int(5 * congestion_level)))

        # Asphalt base + lane markings
        frame = np.full((self.height, self.width, 3), 48, dtype=np.uint8)
        frame[:, :, 1] = 52
        frame[:, :, 2] = 55
        # horizon / sky band
        sky_h = self.height // 5
        frame[:sky_h, :] = (135, 180, 210)
        # dashed center line
        mid = self.width // 2
        for y in range(sky_h + 10, self.height, 24):
            frame[y : y + 10, mid - 2 : mid + 2] = (220, 200, 60)

        detections: list[Detection] = []
        for _ in range(n_vehicles):
            cls = CLASSES[int(self.rng.integers(0, 5))]
            w = float(self.rng.uniform(0.08, 0.18))
            h = float(self.rng.uniform(0.06, 0.14))
            x1 = float(self.rng.uniform(0.05, 0.95 - w))
            y1 = float(self.rng.uniform(0.25, 0.85 - h))
            conf = float(self.rng.uniform(0.6, 0.99))
            detections.append(Detection(cls, conf, (x1, y1, x1 + w, y1 + h)))
            self._paint_box(frame, x1, y1, x1 + w, y1 + h, color=(40, 90, 200))

        for _ in range(n_pedestrians):
            w, h = 0.03, 0.10
            x1 = float(self.rng.uniform(0.05, 0.92))
            y1 = float(self.rng.uniform(0.45, 0.88))
            conf = float(self.rng.uniform(0.55, 0.97))
            detections.append(
                Detection("pedestrian", conf, (x1, y1, x1 + w, y1 + h))
            )
            self._paint_box(frame, x1, y1, x1 + w, y1 + h, color=(200, 120, 60))

        idx = self._index
        self._index += 1
        return SimulatedFrame(
            frame=frame,
            ground_truth=detections,
            camera_id=self.camera_id,
            corridor_id=self.corridor_id,
            frame_index=idx,
        )

    def _paint_box(
        self,
        frame: np.ndarray,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: tuple[int, int, int],
    ) -> None:
        h, w = frame.shape[:2]
        ix1, iy1 = int(x1 * w), int(y1 * h)
        ix2, iy2 = int(x2 * w), int(y2 * h)
        frame[iy1:iy2, ix1:ix2] = color
