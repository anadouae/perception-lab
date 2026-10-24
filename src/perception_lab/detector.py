"""Traffic scene analyzer: turns camera-frame tensors into congestion / detection events."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import numpy as np


# Public-space object classes used for traffic congestion analytics
CLASSES = ("car", "bus", "truck", "motorcycle", "bicycle", "pedestrian")


@dataclass(frozen=True)
class Detection:
    """Single object detection in a traffic camera frame."""

    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]  # normalized 0–1

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox_xyxy": [round(v, 4) for v in self.bbox_xyxy],
        }


@dataclass
class DetectionEvent:
    """JSON-serializable Kafka event for Urban Pulse (`urban.perception.detections`)."""

    camera_id: str
    corridor_id: str
    frame_index: int
    timestamp: str
    vehicle_count: int
    pedestrian_count: int
    congestion_score: float
    detections: list[dict[str, Any]] = field(default_factory=list)
    provenance: str = "SIMULATED"
    event_id: str = field(default_factory=lambda: str(uuid4()))
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrafficSceneAnalyzer:
    """
    Lightweight CPU analytics over RGB frames.

    Designed for smart-city traffic congestion and public-space occupancy
    metrics — not surveillance tooling. Frames may be simulated or sourced
    from authorized municipal camera feeds.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.45,
        density_weight: float = 0.7,
        motion_weight: float = 0.3,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        self.confidence_threshold = confidence_threshold
        self.density_weight = density_weight
        self.motion_weight = motion_weight
        self._prev_gray: np.ndarray | None = None

    def analyze_frame(
        self,
        frame: np.ndarray,
        *,
        camera_id: str,
        corridor_id: str,
        frame_index: int,
        ground_truth: list[Detection] | None = None,
    ) -> DetectionEvent:
        """
        Analyze one HxWx3 uint8 RGB frame and emit a DetectionEvent.

        When ``ground_truth`` is provided (simulator path), detections are
        filtered by confidence. Otherwise a density heuristic produces
        synthetic counts suitable for pipeline demos.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"expected HxWx3 RGB frame, got shape {frame.shape}")

        if ground_truth is not None:
            detections = [
                d
                for d in ground_truth
                if d.confidence >= self.confidence_threshold and d.class_name in CLASSES
            ]
        else:
            detections = self._heuristic_detections(frame)

        vehicle_classes = {"car", "bus", "truck", "motorcycle", "bicycle"}
        vehicle_count = sum(1 for d in detections if d.class_name in vehicle_classes)
        pedestrian_count = sum(1 for d in detections if d.class_name == "pedestrian")
        congestion = self._congestion_score(frame, vehicle_count, pedestrian_count)

        return DetectionEvent(
            camera_id=camera_id,
            corridor_id=corridor_id,
            frame_index=frame_index,
            timestamp=datetime.now(timezone.utc).isoformat(),
            vehicle_count=vehicle_count,
            pedestrian_count=pedestrian_count,
            congestion_score=round(congestion, 4),
            detections=[d.to_dict() for d in detections],
        )

    def _heuristic_detections(self, frame: np.ndarray) -> list[Detection]:
        """Density-based synthetic detections when no ground truth is supplied."""
        gray = frame.mean(axis=2)
        edge_energy = float(np.std(np.diff(gray, axis=1)))
        # Map edge energy into a plausible vehicle count for a city corridor
        n_vehicles = int(np.clip(edge_energy / 8.0, 0, 25))
        n_pedestrians = int(np.clip(edge_energy / 20.0, 0, 10))
        detections: list[Detection] = []
        rng = np.random.default_rng(int(edge_energy * 1000) % (2**32))
        for _ in range(n_vehicles):
            cls = CLASSES[int(rng.integers(0, 5))]
            x1, y1 = float(rng.uniform(0.05, 0.7)), float(rng.uniform(0.2, 0.7))
            detections.append(
                Detection(
                    class_name=cls,
                    confidence=float(rng.uniform(0.55, 0.98)),
                    bbox_xyxy=(x1, y1, min(x1 + 0.15, 0.95), min(y1 + 0.12, 0.95)),
                )
            )
        for _ in range(n_pedestrians):
            x1, y1 = float(rng.uniform(0.05, 0.8)), float(rng.uniform(0.4, 0.85))
            detections.append(
                Detection(
                    class_name="pedestrian",
                    confidence=float(rng.uniform(0.5, 0.95)),
                    bbox_xyxy=(x1, y1, min(x1 + 0.05, 0.98), min(y1 + 0.15, 0.98)),
                )
            )
        return detections

    def _congestion_score(
        self, frame: np.ndarray, vehicle_count: int, pedestrian_count: int
    ) -> float:
        """
        Congestion in [0, 1]: blend of object density and frame-to-frame motion.

        High vehicle density + low motion ≈ stopped traffic (high congestion).
        """
        density = min(1.0, (vehicle_count + 0.3 * pedestrian_count) / 20.0)
        gray = frame.mean(axis=2).astype(np.float32)
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            motion = 0.5
        else:
            motion = float(np.mean(np.abs(gray - self._prev_gray)) / 255.0)
            motion = float(np.clip(motion * 4.0, 0.0, 1.0))
        self._prev_gray = gray
        # Low motion with high density → congestion rises
        score = self.density_weight * density + self.motion_weight * (1.0 - motion) * density
        return float(np.clip(score, 0.0, 1.0))


def f1_score(y_true: list[str], y_pred: list[str], labels: tuple[str, ...] | None = None) -> float:
    """Macro-F1 over labels that appear in either list (demo metric for evals)."""
    if not y_true and not y_pred:
        return 1.0
    label_set = labels if labels is not None else tuple(sorted(set(y_true) | set(y_pred)))
    if not label_set:
        return 0.0
    scores: list[float] = []
    for label in label_set:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        if len(y_pred) > len(y_true):
            fp += sum(1 for p in y_pred[len(y_true) :] if p == label)
        elif len(y_true) > len(y_pred):
            fn += sum(1 for t in y_true[len(y_pred) :] if t == label)
        # Skip labels with no support in either list
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return float(sum(scores) / len(scores)) if scores else 0.0
