"""
APT: Adaptive Plugin for Test-time Adaptation via Temporal Consistency

This module implements test-time adaptation for object detection under
continual domain shift using temporal consistency via Kalman filter tracking.
"""

from .config import APTConfig
from .engine import APTEngine
from .tracker import TemporalTracker, KalmanBBoxTracker
