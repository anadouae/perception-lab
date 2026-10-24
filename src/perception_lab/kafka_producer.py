"""Kafka producer for Urban Pulse detection events."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Protocol

from perception_lab.detector import DetectionEvent

logger = logging.getLogger(__name__)

DEFAULT_TOPIC = "urban.perception.detections"


class EventSink(Protocol):
    """Abstraction so tests can use an in-memory sink without a broker."""

    def publish(self, event: DetectionEvent, *, topic: str = DEFAULT_TOPIC) -> None: ...

    def close(self) -> None: ...


class InMemorySink:
    """Collects events for unit tests and offline demos."""

    def __init__(self) -> None:
        self.records: list[tuple[str, dict[str, Any]]] = []

    def publish(self, event: DetectionEvent, *, topic: str = DEFAULT_TOPIC) -> None:
        self.records.append((topic, event.to_dict()))

    def close(self) -> None:
        return None


class KafkaDetectionProducer:
    """
    Publishes DetectionEvent JSON to Kafka (topic ``urban.perception.detections``).

    Uses ``confluent-kafka``. When ``bootstrap_servers`` is empty or the client
    cannot be created, falls back to :class:`InMemorySink` so local demos work
    without a broker.
    """

    def __init__(
        self,
        bootstrap_servers: str | None = None,
        topic: str = DEFAULT_TOPIC,
        client_id: str = "perception-lab",
        *,
        force_memory: bool = False,
    ) -> None:
        self.topic = topic
        self.bootstrap_servers = (
            bootstrap_servers
            if bootstrap_servers is not None
            else os.getenv("KAFKA_BOOTSTRAP_SERVERS", "")
        )
        self._memory = InMemorySink()
        self._producer = None

        if force_memory or not self.bootstrap_servers.strip():
            logger.info("KafkaDetectionProducer using InMemorySink (no brokers configured)")
            return

        try:
            from confluent_kafka import Producer  # type: ignore

            self._producer = Producer(
                {
                    "bootstrap.servers": self.bootstrap_servers,
                    "client.id": client_id,
                    "acks": "all",
                    "linger.ms": 5,
                }
            )
            logger.info(
                "KafkaDetectionProducer connected to %s topic=%s",
                self.bootstrap_servers,
                self.topic,
            )
        except Exception as exc:  # pragma: no cover - broker unavailable
            logger.warning("Kafka unavailable (%s); falling back to InMemorySink", exc)
            self._producer = None

    def publish(self, event: DetectionEvent, *, topic: str | None = None) -> None:
        target = topic or self.topic
        payload = event.to_dict()
        if self._producer is None:
            self._memory.publish(event, topic=target)
            return

        key = event.camera_id.encode("utf-8")
        value = json.dumps(payload, separators=(",", ":")).encode("utf-8")

        def _cb(err: Any, msg: Any) -> None:
            if err is not None:
                logger.error("Kafka delivery failed: %s", err)
            else:
                logger.debug(
                    "delivered topic=%s partition=%s offset=%s",
                    msg.topic(),
                    msg.partition(),
                    msg.offset(),
                )

        self._producer.produce(target, key=key, value=value, callback=_cb)
        self._producer.poll(0)

    def flush(self, timeout: float = 5.0) -> None:
        if self._producer is not None:
            self._producer.flush(timeout)

    def close(self) -> None:
        self.flush()
        self._memory.close()

    @property
    def buffered(self) -> list[tuple[str, dict[str, Any]]]:
        """Events captured when running without a live broker."""
        return list(self._memory.records)
