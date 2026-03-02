"""
WiFi packet capture and monitoring.

Captures WiFi frames in monitor mode and feeds them to threat detectors.
"""

import asyncio
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ...config import settings
from .deauth_detector import DeauthDetector
from .rogue_ap_detector import RogueAPDetector

logger = logging.getLogger("atlas.security.wireless.monitor")

MIN_SIGNAL_DBM = -100

sniff = None
Dot11 = None
Dot11Beacon = None
Dot11Elt = None
Dot11Deauth = None
_SCAPY_IMPORT_ERROR: Exception | None = None


def _ensure_scapy() -> None:
    """Import scapy lazily to avoid privileged socket access during collection."""
    global sniff, Dot11, Dot11Beacon, Dot11Elt, Dot11Deauth, _SCAPY_IMPORT_ERROR

    if sniff is not None and Dot11 is not None:
        return

    try:
        from scapy.all import (  # type: ignore
            Dot11 as _Dot11,
            Dot11Beacon as _Dot11Beacon,
            Dot11Deauth as _Dot11Deauth,
            Dot11Elt as _Dot11Elt,
            sniff as _sniff,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        _SCAPY_IMPORT_ERROR = exc
        raise RuntimeError(
            "Scapy is unavailable for wireless monitoring in this environment."
        ) from exc

    sniff = _sniff
    Dot11 = _Dot11
    Dot11Beacon = _Dot11Beacon
    Dot11Elt = _Dot11Elt
    Dot11Deauth = _Dot11Deauth

class WirelessMonitor:
    """
    WiFi packet capture and monitoring system.
    
    Captures packets in monitor mode and processes them for threats.
    """
    
    def __init__(self):
        self._interface = settings.security.wireless_interface
        self._channels = settings.security.wireless_channels
        self._hop_interval = settings.security.channel_hop_interval
        self._running = False
        self._capture_task: Optional[asyncio.Task] = None
        self._hop_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._deauth_detector = DeauthDetector()
        self._rogue_ap_detector = RogueAPDetector()
        self._packet_count = 0
        self._current_channel = 0
        
    def _check_interface_exists(self) -> bool:
        """Check if wireless interface exists."""
        try:
            result = subprocess.run(
                ["iwconfig", self._interface],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception as e:
            logger.error("Failed to check interface: %s", e)
            return False
            
    def _set_channel(self, channel: int) -> bool:
        """Set WiFi channel for monitoring."""
        try:
            result = subprocess.run(
                ["iwconfig", self._interface, "channel", str(channel)],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self._current_channel = channel
                return True
            logger.error("Failed to set channel %d: %s", channel, result.stderr)
            return False
        except Exception as e:
            logger.error("Exception setting channel: %s", e)
            return False
            
    def _process_packet(self, packet) -> None:
        """Process captured WiFi packet."""
        _ensure_scapy()

        if not packet.haslayer(Dot11):
            return
            
        self._packet_count += 1
        if self._packet_count % 1000 == 0:
            logger.debug("Processed %d packets", self._packet_count)
            
        try:
            dot11 = packet.getlayer(Dot11)
            
            if packet.haslayer(Dot11Deauth):
                src_mac = dot11.addr2
                dst_mac = dot11.addr1
                if src_mac and dst_mac:
                    if self._deauth_detector.process_deauth_frame(src_mac, dst_mac):
                        if self._loop and self._loop.is_running():
                            asyncio.run_coroutine_threadsafe(
                                self._alert_deauth_attack(src_mac, dst_mac),
                                self._loop
                            )
                        
            elif packet.haslayer(Dot11Beacon):
                bssid = dot11.addr3
                ssid = None
                channel = self._current_channel
                signal = packet.dBm_AntSignal if hasattr(packet, "dBm_AntSignal") else MIN_SIGNAL_DBM
                
                elt_layer = packet.getlayer(Dot11Elt)
                while elt_layer:
                    if elt_layer.ID == 0:
                        ssid = elt_layer.info.decode("utf-8", errors="ignore")
                        break
                    elt_layer = elt_layer.payload.getlayer(Dot11Elt)
                    
                if ssid and bssid:
                    alert_data = self._rogue_ap_detector.process_beacon(
                        bssid, ssid, channel, signal
                    )
                    if alert_data and self._loop and self._loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            self._alert_rogue_ap(alert_data),
                            self._loop
                        )
                        
        except Exception as e:
            logger.debug("Error processing packet: %s", e)
            
    async def _alert_deauth_attack(self, src_mac: str, dst_mac: str) -> None:
        """Generate alert for deauth attack."""
        from ...alerts import SecurityAlertEvent
        from ...alerts.manager import get_alert_manager
        
        alert_manager = get_alert_manager()
        await alert_manager.process_event(SecurityAlertEvent(
            source_id=self._interface,
            timestamp=datetime.now(timezone.utc),
            detection_type="deauth_attack",
            label="WiFi Deauth Attack",
            confidence=1.0,
            metadata={
                "attacker_mac": src_mac,
                "target_mac": dst_mac,
                "channel": self._current_channel,
                "threshold": settings.security.deauth_threshold,
                "message": f"WiFi deauth attack: {src_mac} targeting {dst_mac}"
            }
        ))
        
    async def _alert_rogue_ap(self, alert_data: dict) -> None:
        """Generate alert for rogue AP."""
        from ...alerts import SecurityAlertEvent
        from ...alerts.manager import get_alert_manager
        
        alert_manager = get_alert_manager()
        await alert_manager.process_event(SecurityAlertEvent(
            source_id=self._interface,
            timestamp=datetime.now(timezone.utc),
            detection_type="rogue_ap",
            label=f"Rogue AP: {alert_data['ssid']}",
            confidence=1.0,
            metadata={
                **alert_data,
                "message": f"Rogue AP detected: {alert_data['ssid']} ({alert_data['bssid']})"
            }
        ))
            
    async def _channel_hopper(self) -> None:
        """Hop between configured channels."""
        channel_idx = 0
        while self._running:
            channel = self._channels[channel_idx]
            self._set_channel(channel)
            logger.debug("Switched to channel %d", channel)
            
            channel_idx = (channel_idx + 1) % len(self._channels)
            await asyncio.sleep(self._hop_interval)
            
    def _start_packet_capture(self) -> None:
        """Start Scapy packet capture (runs in thread)."""
        _ensure_scapy()

        logger.info("Starting packet capture on %s", self._interface)
        sniff(
            iface=self._interface,
            prn=self._process_packet,
            store=0,
            stop_filter=lambda _: not self._running
        )
        logger.info("Packet capture stopped")
            
    async def start(self) -> None:
        """Start wireless monitoring."""
        if self._running:
            logger.warning("Wireless monitor already running")
            return
            
        if not self._check_interface_exists():
            raise RuntimeError(
                f"Wireless interface {self._interface} not found or not in monitor mode"
            )
            
        logger.info("Starting wireless monitor on %s", self._interface)
        self._running = True
        
        if len(self._channels) > 1:
            self._hop_task = asyncio.create_task(self._channel_hopper())
        elif self._channels:
            self._set_channel(self._channels[0])
            
        self._loop = asyncio.get_running_loop()
        self._capture_task = self._loop.run_in_executor(None, self._start_packet_capture)
            
        logger.info("Wireless monitor started on channel(s): %s", self._channels)
        
    async def stop(self) -> None:
        """Stop wireless monitoring."""
        if not self._running:
            return
            
        logger.info("Stopping wireless monitor")
        self._running = False
        
        if self._hop_task:
            self._hop_task.cancel()
            try:
                await self._hop_task
            except asyncio.CancelledError:
                pass
            self._hop_task = None
            
        if self._capture_task:
            try:
                await asyncio.wait_for(self._capture_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Packet capture did not stop gracefully")
            self._capture_task = None
            
        logger.info("Wireless monitor stopped")
