"""
Configuration management for Atlas Vision.

Uses Pydantic Settings for environment variable support.
Mirrors atlas_brain configuration pattern.
"""

import logging
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("atlas.vision.config")


class ServerConfig(BaseSettings):
    """Server configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_VISION_")

    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=5002, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")


class MQTTConfig(BaseSettings):
    """MQTT configuration for telemetry and events."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_VISION_MQTT_")

    enabled: bool = Field(default=False, description="Enable MQTT")
    host: str = Field(default="localhost", description="MQTT broker host")
    port: int = Field(default=1883, description="MQTT broker port")
    username: Optional[str] = Field(default=None, description="MQTT username")
    password: Optional[str] = Field(default=None, description="MQTT password")


class CameraConfig(BaseSettings):
    """Camera configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_VISION_CAMERA_")

    discovery_enabled: bool = Field(default=True, description="Auto-discover cameras")
    default_timeout: float = Field(default=10.0, description="Camera timeout")
    snapshot_quality: int = Field(default=85, description="JPEG quality 1-100")


class DiscoveryConfig(BaseSettings):
    """Discovery configuration for node announcement."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_VISION_DISCOVERY_")

    enabled: bool = Field(default=True, description="Enable mDNS announcement")
    node_name: str = Field(default="atlas-vision", description="Node name")
    announce_interval: int = Field(default=60, description="Announce interval seconds")


class DetectionConfig(BaseSettings):
    """Object detection and tracking configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_VISION_DETECTION_")

    enabled: bool = Field(default=True, description="Enable object detection")
    model: str = Field(default="yolov8n.pt", description="YOLO model name")
    confidence_threshold: float = Field(default=0.5, description="Min confidence 0-1")
    device: str = Field(default="auto", description="Device: auto, cuda, cpu")
    fps: float = Field(default=10.0, description="Detection frames per second")
    track_classes: list[str] = Field(
        default=["person", "car", "truck", "dog", "cat", "bicycle", "motorcycle"],
        description="Classes to track"
    )
    max_track_history: int = Field(default=100, description="Track history points to keep")
    track_timeout: float = Field(default=5.0, description="Seconds before track is lost")


class RecognitionConfig(BaseSettings):
    """Person recognition configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_VISION_RECOGNITION_")

    enabled: bool = Field(default=True, description="Enable recognition")
    face_threshold: float = Field(default=0.6, description="Face match threshold")
    gait_threshold: float = Field(default=0.5, description="Gait match threshold")
    auto_enroll_unknown: bool = Field(default=True, description="Auto-enroll unknown")
    use_averaged: bool = Field(default=True, description="Use averaged embeddings")
    gait_sequence_length: int = Field(default=60, description="Frames for gait")
    mediapipe_detection_confidence: float = Field(default=0.5, description="Pose detect")
    mediapipe_tracking_confidence: float = Field(default=0.5, description="Pose track")
    track_timeout: float = Field(default=10.0, description="Track timeout seconds")
    max_tracked_persons: int = Field(default=10, description="Max tracked persons")


class PresenceConfig(BaseSettings):
    """Presence tracking configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_VISION_PRESENCE_")

    enabled: bool = Field(default=True, description="Enable presence tracking")
    espresense_enabled: bool = Field(default=True, description="Use ESPresense BLE")
    camera_enabled: bool = Field(default=True, description="Use camera detection")


class Settings(BaseSettings):
    """Main settings aggregator."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_VISION_")

    # Sub-configurations
    server: ServerConfig = Field(default_factory=ServerConfig)
    mqtt: MQTTConfig = Field(default_factory=MQTTConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    recognition: RecognitionConfig = Field(default_factory=RecognitionConfig)
    presence: PresenceConfig = Field(default_factory=PresenceConfig)

    # Logging
    log_level: str = Field(default="INFO", description="Log level")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
