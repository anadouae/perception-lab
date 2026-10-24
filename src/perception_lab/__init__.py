"""Perception Lab — urban traffic scene analytics for the Urban Pulse ecosystem."""

from perception_lab.detector import TrafficSceneAnalyzer, DetectionEvent
from perception_lab.simulator import FrameSimulator

__version__ = "0.1.0"
__all__ = ["TrafficSceneAnalyzer", "DetectionEvent", "FrameSimulator", "__version__"]
