"""
APT: Adaptive Plugin for Test-time Adaptation via Temporal Consistency

This module implements test-time adaptation for object detection under
continual domain shift using temporal consistency via Kalman filter tracking.
"""

from .apt_config import APTConfig
from .apt_plugin import APTPlugin
from .kalman_tracker import TemporalTracker, KalmanBBoxTracker
