"""CLI: simulate traffic frames and publish detection events."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from perception_lab.detector import TrafficSceneAnalyzer
from perception_lab.kafka_producer import DEFAULT_TOPIC, KafkaDetectionProducer
from perception_lab.simulator import FrameSimulator


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="perception-lab",
        description="Simulate urban traffic camera analytics and publish JSON events.",
    )
    p.add_argument("--frames", type=int, default=10, help="Number of frames to process")
    p.add_argument("--congestion", type=float, default=0.45, help="Simulated congestion [0-1]")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap", default="", help="Kafka bootstrap servers (optional)")
    p.add_argument("--topic", default=DEFAULT_TOPIC)
    p.add_argument("--sleep-ms", type=int, default=0, help="Delay between frames")
    p.add_argument("--jsonl", action="store_true", help="Print events as JSON lines to stdout")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    sim = FrameSimulator(seed=args.seed)
    analyzer = TrafficSceneAnalyzer()
    producer = KafkaDetectionProducer(
        bootstrap_servers=args.bootstrap or None,
        topic=args.topic,
        force_memory=not bool(args.bootstrap),
    )

    for _ in range(args.frames):
        sf = sim.next_frame(congestion_level=args.congestion)
        event = analyzer.analyze_frame(
            sf.frame,
            camera_id=sf.camera_id,
            corridor_id=sf.corridor_id,
            frame_index=sf.frame_index,
            ground_truth=sf.ground_truth,
        )
        producer.publish(event)
        if args.jsonl:
            print(json.dumps(event.to_dict(), separators=(",", ":")))
        if args.sleep_ms:
            time.sleep(args.sleep_ms / 1000.0)

    producer.flush()
    buffered = producer.buffered
    if buffered and not args.jsonl:
        print(
            f"Processed {args.frames} frames; "
            f"{len(buffered)} events buffered (topic={args.topic}).",
            file=sys.stderr,
        )
    producer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
