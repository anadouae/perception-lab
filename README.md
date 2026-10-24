# Perception Lab

**Urban traffic scene analytics** for the [Urban Pulse](https://github.com/anadouae/urban-pulse-intelligence) smart-city twin.

Simulates municipal traffic-camera frames, extracts vehicle / pedestrian detections and a **congestion score**, then publishes JSON events to Kafka topic `urban.perception.detections`.

> Portfolio project — PFE 2027 · ENSA Tétouan · Douae El Assal (`anadouae`).  
> Analytics for **traffic congestion** and **public-space occupancy** only. Frames are simulated (or would come from authorized city feeds). Not surveillance or access tooling.

## Architecture role

```text
Simulated camera frames
        │
        ▼
  Perception Lab  ──JSON──►  Kafka: urban.perception.detections
        │
        ▼
   Urban Pulse (dashboard / twin)
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
perception-lab --frames 20 --jsonl
```

Without Kafka brokers configured, events are buffered in memory (demo mode).

### Optional Kafka

```bash
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092
perception-lab --frames 50 --bootstrap "$KAFKA_BOOTSTRAP_SERVERS" --jsonl
```

## Event schema (`urban.perception.detections`)

```json
{
  "event_id": "…",
  "schema_version": "1.0",
  "camera_id": "cam-avenue-hassan-01",
  "corridor_id": "corridor-tetouan-centre",
  "frame_index": 0,
  "timestamp": "2026-10-24T00:00:00+00:00",
  "vehicle_count": 6,
  "pedestrian_count": 2,
  "congestion_score": 0.42,
  "detections": [
    {"class": "car", "confidence": 0.91, "bbox_xyxy": [0.1, 0.3, 0.25, 0.42]}
  ],
  "provenance": "SIMULATED"
}
```

## Package layout

| Path | Role |
|------|------|
| `src/perception_lab/simulator.py` | Procedural corridor frames + ground truth |
| `src/perception_lab/detector.py` | Congestion / detection analytics |
| `src/perception_lab/kafka_producer.py` | Kafka producer (in-memory fallback) |
| `src/perception_lab/cli.py` | `perception-lab` CLI |
| `tests/` | pytest suite |

## Docker (optional)

```bash
docker build -t perception-lab .
docker run --rm perception-lab --frames 5 --jsonl
```

## License

MIT
