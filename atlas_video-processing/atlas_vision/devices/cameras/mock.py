"""
Mock camera for testing without real hardware.
"""

import random
from datetime import datetime, timedelta
from typing import Optional

from ...core.constants import DeviceStatus, DetectionType
from ...core.protocols import Detection, MotionEvent
from .base import BaseCameraCapability


class MockCamera(BaseCameraCapability):
    """Mock camera that generates simulated data."""

    def __init__(
        self,
        device_id: str,
        name: str,
        location: str,
        simulate_detections: bool = True,
    ):
        super().__init__(device_id, name, location)
        self._simulate_detections = simulate_detections
        self._status = DeviceStatus.ONLINE
        self._last_motion = datetime.now() - timedelta(minutes=random.randint(5, 60))

    async def connect(self) -> bool:
        """Mock connect - always succeeds."""
        self._status = DeviceStatus.ONLINE
        return True

    async def disconnect(self) -> None:
        """Mock disconnect."""
        self._status = DeviceStatus.OFFLINE

    async def get_current_detections(self) -> list[Detection]:
        """Generate simulated detections."""
        if not self._simulate_detections:
            return []

        detections = []

        # Randomly decide if there's a detection (20% chance)
        if random.random() < 0.2:
            detection_types = [
                (DetectionType.PERSON, ["unknown", "known:Juan", "known:Guest"]),
                (DetectionType.VEHICLE, ["car", "delivery_truck", "motorcycle"]),
                (DetectionType.ANIMAL, ["cat", "dog", "bird"]),
            ]

            det_type, labels = random.choice(detection_types)
            detections.append(
                Detection(
                    camera_id=self._device_id,
                    timestamp=datetime.now(),
                    detection_type=det_type,
                    confidence=random.uniform(0.75, 0.99),
                    label=random.choice(labels),
                )
            )

        return detections

    async def get_motion_events(self, since: Optional[datetime] = None) -> list[MotionEvent]:
        """Generate simulated motion events."""
        if since is None:
            since = datetime.now() - timedelta(hours=1)

        events = []
        current = since

        # Generate 0-5 random events
        num_events = random.randint(0, 5)
        for _ in range(num_events):
            current = current + timedelta(minutes=random.randint(5, 30))
            if current > datetime.now():
                break

            events.append(
                MotionEvent(
                    camera_id=self._device_id,
                    timestamp=current,
                    confidence=random.uniform(0.8, 1.0),
                )
            )

        return events


def create_mock_cameras() -> list[MockCamera]:
    """Create default mock cameras matching atlas_brain expectations."""
    return [
        MockCamera("cam_front_door", "Front Door", "entrance"),
        MockCamera("cam_backyard", "Backyard", "exterior"),
        MockCamera("cam_garage", "Garage", "garage"),
        MockCamera("cam_driveway", "Driveway", "exterior"),
        MockCamera("cam_living_room", "Living Room", "interior"),
        MockCamera("cam_kitchen", "Kitchen", "interior"),
    ]
