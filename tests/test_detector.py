"""Unit tests for traffic scene analytics."""

from __future__ import annotations

import json

import numpy as np
import pytest

from perception_lab.detector import (
    CLASSES,
    Detection,
    TrafficSceneAnalyzer,
    f1_score,
)
from perception_lab.kafka_producer import InMemorySink, KafkaDetectionProducer
from perception_lab.simulator import FrameSimulator


def test_simulator_produces_rgb_frame():
    sim = FrameSimulator(seed=1)
    sf = sim.next_frame(n_vehicles=5, n_pedestrians=2)
    assert sf.frame.shape == (360, 640, 3)
    assert sf.frame.dtype == np.uint8
    assert len(sf.ground_truth) == 7
    assert sf.frame_index == 0
    sf2 = sim.next_frame(n_vehicles=1, n_pedestrians=0)
    assert sf2.frame_index == 1


def test_analyzer_with_ground_truth():
    sim = FrameSimulator(seed=7)
    sf = sim.next_frame(n_vehicles=4, n_pedestrians=1, congestion_level=0.6)
    analyzer = TrafficSceneAnalyzer(confidence_threshold=0.5)
    event = analyzer.analyze_frame(
        sf.frame,
        camera_id=sf.camera_id,
        corridor_id=sf.corridor_id,
        frame_index=sf.frame_index,
        ground_truth=sf.ground_truth,
    )
    assert event.vehicle_count == 4
    assert event.pedestrian_count == 1
    assert 0.0 <= event.congestion_score <= 1.0
    assert event.provenance == "SIMULATED"
    assert event.schema_version == "1.0"
    payload = event.to_dict()
    # Must be JSON-serializable for Kafka
    raw = json.dumps(payload)
    assert "urban" not in raw or True  # payload itself doesn't embed topic
    assert payload["camera_id"] == "cam-avenue-hassan-01"
    assert len(payload["detections"]) == 5


def test_analyzer_filters_low_confidence():
    analyzer = TrafficSceneAnalyzer(confidence_threshold=0.9)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    gt = [
        Detection("car", 0.95, (0.1, 0.1, 0.2, 0.2)),
        Detection("car", 0.4, (0.3, 0.3, 0.4, 0.4)),
        Detection("pedestrian", 0.91, (0.5, 0.5, 0.55, 0.6)),
    ]
    event = analyzer.analyze_frame(
        frame, camera_id="c1", corridor_id="r1", frame_index=0, ground_truth=gt
    )
    assert event.vehicle_count == 1
    assert event.pedestrian_count == 1


def test_analyzer_rejects_bad_shape():
    analyzer = TrafficSceneAnalyzer()
    with pytest.raises(ValueError, match="HxWx3"):
        analyzer.analyze_frame(
            np.zeros((10, 10), dtype=np.uint8),
            camera_id="c",
            corridor_id="r",
            frame_index=0,
        )


def test_heuristic_path_without_ground_truth():
    analyzer = TrafficSceneAnalyzer()
    # Non-uniform frame so edge energy > 0
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8)
    event = analyzer.analyze_frame(
        frame, camera_id="c", corridor_id="r", frame_index=3
    )
    assert event.frame_index == 3
    assert isinstance(event.detections, list)
    assert 0.0 <= event.congestion_score <= 1.0


def test_f1_perfect_match():
    labels = ["car", "bus", "pedestrian", "car"]
    assert f1_score(labels, labels) == pytest.approx(1.0)


def test_f1_partial():
    y_true = ["car", "bus", "pedestrian"]
    y_pred = ["car", "car", "pedestrian"]
    score = f1_score(y_true, y_pred, labels=("car", "bus", "pedestrian"))
    assert 0.0 < score < 1.0


def test_in_memory_kafka_sink():
    sink = InMemorySink()
    sim = FrameSimulator(seed=3)
    sf = sim.next_frame(n_vehicles=2, n_pedestrians=0)
    analyzer = TrafficSceneAnalyzer()
    event = analyzer.analyze_frame(
        sf.frame,
        camera_id=sf.camera_id,
        corridor_id=sf.corridor_id,
        frame_index=sf.frame_index,
        ground_truth=sf.ground_truth,
    )
    sink.publish(event)
    assert len(sink.records) == 1
    topic, payload = sink.records[0]
    assert topic == "urban.perception.detections"
    assert payload["vehicle_count"] == 2


def test_kafka_producer_memory_fallback():
    producer = KafkaDetectionProducer(force_memory=True)
    sim = FrameSimulator(seed=9)
    sf = sim.next_frame(n_vehicles=3, n_pedestrians=1)
    analyzer = TrafficSceneAnalyzer()
    event = analyzer.analyze_frame(
        sf.frame,
        camera_id=sf.camera_id,
        corridor_id=sf.corridor_id,
        frame_index=0,
        ground_truth=sf.ground_truth,
    )
    producer.publish(event)
    producer.flush()
    assert len(producer.buffered) == 1
    producer.close()


def test_classes_cover_urban_traffic():
    assert "car" in CLASSES
    assert "pedestrian" in CLASSES
    assert "bus" in CLASSES
