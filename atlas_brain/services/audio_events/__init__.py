"""Audio event detection services."""

from .monitor import (
    AudioMonitor,
    AudioMonitorConfig,
    MicrophoneMonitor,
    MonitoredEvent,
    get_audio_monitor,
    set_audio_monitor,
)
from .yamnet import INTERESTING_EVENTS, YAMNetClassifier, YAMNetTensorFlow

__all__ = [
    # Classifiers
    "YAMNetClassifier",
    "YAMNetTensorFlow",
    "INTERESTING_EVENTS",
    # Monitoring
    "AudioMonitor",
    "AudioMonitorConfig",
    "MicrophoneMonitor",
    "MonitoredEvent",
    "get_audio_monitor",
    "set_audio_monitor",
]
