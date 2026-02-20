"""Tests for the discovery module: scanners, SSDP identification, and service orchestration."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from atlas_brain.discovery.scanners.base import BaseScanner, ScanResult
from atlas_brain.discovery.scanners.ssdp import (
    DEVICE_PATTERNS,
    SSDP_ADDR,
    SSDP_PORT,
    SSDPScanner,
)
from atlas_brain.discovery.service import (
    DEVICE_TYPE_TO_ASSET_TYPE,
    DiscoveryService,
    _discovery_service,
    get_discovery_service,
)
from atlas_brain.storage.models import DiscoveredDevice


# ---------------------------------------------------------------------------
# TestScanResult
# ---------------------------------------------------------------------------


class TestScanResult:
    """Tests for ScanResult dataclass and its generate_device_id method."""

    def test_generate_device_id_with_device_type(self):
        result = ScanResult(host="192.168.1.1", protocol="ssdp", device_type="roku")
        assert result.generate_device_id() == "roku.192_168_1_1"

    def test_generate_device_id_without_device_type(self):
        result = ScanResult(host="10.0.0.1", protocol="mdns")
        assert result.generate_device_id() == "mdns.10_0_0_1"

    def test_generate_device_id_special_chars_in_host(self):
        # IPv4 dots get replaced with underscores
        result = ScanResult(host="172.16.254.99", protocol="ssdp", device_type="router")
        assert result.generate_device_id() == "router.172_16_254_99"

    def test_generate_device_id_hostname_no_dots(self):
        # Hostname without dots passes through unchanged
        result = ScanResult(host="localhost", protocol="mdns")
        assert result.generate_device_id() == "mdns.localhost"

    def test_default_empty_collections(self):
        result = ScanResult(host="10.0.0.1", protocol="ssdp")
        assert result.headers == {}
        assert result.services == []
        assert result.raw_data == {}
        assert result.device_type is None
        assert result.name is None
        assert result.manufacturer is None
        assert result.model is None
        assert result.port is None
        assert result.mac_address is None

    def test_all_fields_populated(self):
        result = ScanResult(
            host="192.168.1.50",
            protocol="ssdp",
            device_type="roku",
            name="Living Room Roku",
            manufacturer="Roku Inc",
            model="Ultra 4800",
            port=8060,
            mac_address="AA:BB:CC:DD:EE:FF",
            headers={"SERVER": "Roku/9.4"},
            services=["roku:ecp"],
            raw_data={"raw_key": "raw_value"},
        )
        assert result.host == "192.168.1.50"
        assert result.protocol == "ssdp"
        assert result.device_type == "roku"
        assert result.name == "Living Room Roku"
        assert result.manufacturer == "Roku Inc"
        assert result.model == "Ultra 4800"
        assert result.port == 8060
        assert result.mac_address == "AA:BB:CC:DD:EE:FF"
        assert result.headers == {"SERVER": "Roku/9.4"}
        assert result.services == ["roku:ecp"]
        assert result.raw_data == {"raw_key": "raw_value"}


# ---------------------------------------------------------------------------
# TestSSDPScanner
# ---------------------------------------------------------------------------


class TestSSDPScanner:
    """Tests for SSDPScanner protocol name, device identification, and response parsing."""

    def setup_method(self):
        self.scanner = SSDPScanner()

    def test_protocol_name(self):
        assert self.scanner.protocol_name == "ssdp"

    def test_identify_roku(self):
        result = ScanResult(
            host="192.168.1.10",
            protocol="ssdp",
            services=["roku:ecp"],
        )
        assert self.scanner.identify_device_type(result) == "roku"

    def test_identify_roku_via_name(self):
        result = ScanResult(
            host="192.168.1.10",
            protocol="ssdp",
            name="Roku Ultra",
        )
        assert self.scanner.identify_device_type(result) == "roku"

    def test_identify_chromecast(self):
        result = ScanResult(
            host="192.168.1.20",
            protocol="ssdp",
            name="Chromecast-XYZ",
        )
        assert self.scanner.identify_device_type(result) == "chromecast"

    def test_identify_chromecast_via_google_cast(self):
        result = ScanResult(
            host="192.168.1.21",
            protocol="ssdp",
            headers={"DEVICE": "google cast receiver"},
        )
        assert self.scanner.identify_device_type(result) == "chromecast"

    def test_identify_smart_tv_samsung(self):
        result = ScanResult(
            host="192.168.1.30",
            protocol="ssdp",
            name="Samsung TV QN85A",
        )
        assert self.scanner.identify_device_type(result) == "smart_tv"

    def test_identify_smart_tv_lg(self):
        result = ScanResult(
            host="192.168.1.31",
            protocol="ssdp",
            name="LG TV OLED55",
        )
        assert self.scanner.identify_device_type(result) == "smart_tv"

    def test_identify_upnp_media_renderer_matches_smart_tv(self):
        # smart_tv patterns include "urn:schemas-upnp-org:device:MediaRenderer"
        # which is checked before the media_renderer patterns in dict order,
        # so a full UPnP URN service matches smart_tv first.
        result = ScanResult(
            host="192.168.1.40",
            protocol="ssdp",
            services=["urn:schemas-upnp-org:device:MediaRenderer:1"],
        )
        assert self.scanner.identify_device_type(result) == "smart_tv"

    def test_identify_media_renderer_without_urn(self):
        # When the text contains "MediaRenderer" but NOT the full
        # "urn:schemas-upnp-org:device:MediaRenderer" prefix, the smart_tv
        # URN pattern does not match, so the media_renderer pattern wins.
        result = ScanResult(
            host="192.168.1.41",
            protocol="ssdp",
            name="Custom MediaRenderer Box",
        )
        assert self.scanner.identify_device_type(result) == "media_renderer"

    def test_identify_router(self):
        result = ScanResult(
            host="192.168.1.1",
            protocol="ssdp",
            services=["urn:schemas-upnp-org:device:InternetGatewayDevice:1"],
        )
        assert self.scanner.identify_device_type(result) == "router"

    def test_identify_speaker_sonos(self):
        result = ScanResult(
            host="192.168.1.50",
            protocol="ssdp",
            name="Sonos Speaker One",
        )
        assert self.scanner.identify_device_type(result) == "speaker"

    def test_identify_unknown_device(self):
        # A device with no matching patterns falls back to "upnp_device"
        result = ScanResult(
            host="192.168.1.99",
            protocol="ssdp",
            name="Generic Appliance",
            manufacturer="Acme Corp",
            model="X100",
            services=["some:random:service"],
            headers={"SERVER": "FooOS/1.0"},
        )
        assert self.scanner.identify_device_type(result) == "upnp_device"

    def test_parse_response_basic(self):
        response = (
            "HTTP/1.1 200 OK\r\n"
            "ST: upnp:rootdevice\r\n"
            "LOCATION: http://192.168.1.10:8060/\r\n"
            "SERVER: Roku/9.4 UPnP/1.0\r\n"
            "USN: uuid:roku:ecp:P0A0123456\r\n"
            "CACHE-CONTROL: max-age=1800\r\n"
            "\r\n"
        )
        parsed = self.scanner._parse_response(response)

        # Top-level extracted fields
        assert parsed["st"] == "upnp:rootdevice"
        assert parsed["location"] == "http://192.168.1.10:8060/"
        assert parsed["server"] == "Roku/9.4 UPnP/1.0"
        assert parsed["usn"] == "uuid:roku:ecp:P0A0123456"

        # All headers stored upper-case
        assert parsed["headers"]["ST"] == "upnp:rootdevice"
        assert parsed["headers"]["LOCATION"] == "http://192.168.1.10:8060/"
        assert parsed["headers"]["SERVER"] == "Roku/9.4 UPnP/1.0"
        assert parsed["headers"]["USN"] == "uuid:roku:ecp:P0A0123456"
        assert parsed["headers"]["CACHE-CONTROL"] == "max-age=1800"

    def test_parse_response_empty(self):
        # Status line only, no headers
        response = "HTTP/1.1 200 OK\r\n\r\n"
        parsed = self.scanner._parse_response(response)
        assert parsed["headers"] == {}
        assert "st" not in parsed
        assert "location" not in parsed

    def test_parse_response_missing_value(self):
        # Header line without a colon is ignored
        response = "HTTP/1.1 200 OK\r\nNO_COLON_LINE\r\nST: ssdp:all\r\n\r\n"
        parsed = self.scanner._parse_response(response)
        assert parsed["headers"]["ST"] == "ssdp:all"
        assert parsed["st"] == "ssdp:all"
        # NO_COLON_LINE should not appear in headers
        assert "NO_COLON_LINE" not in parsed["headers"]

    def test_identify_priority_roku_over_media_renderer(self):
        # DEVICE_PATTERNS is iterated in insertion order.
        # "roku" comes before "media_renderer", so if both match, roku wins.
        result = ScanResult(
            host="192.168.1.60",
            protocol="ssdp",
            name="Roku MediaRenderer",
            services=["roku:ecp", "urn:schemas-upnp-org:device:MediaRenderer:1"],
        )
        assert self.scanner.identify_device_type(result) == "roku"


# ---------------------------------------------------------------------------
# TestBaseScanner
# ---------------------------------------------------------------------------


class TestBaseScanner:
    """Tests for BaseScanner default method behavior."""

    @pytest.mark.asyncio
    async def test_is_available_default_returns_true(self):

        class StubScanner(BaseScanner):
            @property
            def protocol_name(self) -> str:
                return "stub"

            async def scan(self, timeout=5.0):
                return []

        scanner = StubScanner()
        assert await scanner.is_available() is True

    def test_identify_device_type_default_returns_none(self):

        class StubScanner(BaseScanner):
            @property
            def protocol_name(self) -> str:
                return "stub"

            async def scan(self, timeout=5.0):
                return []

        scanner = StubScanner()
        result = ScanResult(host="10.0.0.1", protocol="stub")
        assert scanner.identify_device_type(result) is None


# ---------------------------------------------------------------------------
# TestDiscoveryService
# ---------------------------------------------------------------------------


class TestDiscoveryService:
    """Tests for DiscoveryService initialization, scanning, and error handling."""

    @patch("atlas_brain.discovery.service.settings")
    def test_init_scanners_ssdp_only(self, mock_settings):
        mock_settings.discovery.ssdp_enabled = True
        mock_settings.discovery.mdns_enabled = False
        svc = DiscoveryService()
        assert len(svc._scanners) == 1
        assert svc._scanners[0].protocol_name == "ssdp"

    @patch("atlas_brain.discovery.service.settings")
    def test_init_scanners_both(self, mock_settings):
        mock_settings.discovery.ssdp_enabled = True
        mock_settings.discovery.mdns_enabled = True
        svc = DiscoveryService()
        assert len(svc._scanners) == 2
        protocol_names = {s.protocol_name for s in svc._scanners}
        assert "ssdp" in protocol_names
        assert "mdns" in protocol_names

    @patch("atlas_brain.discovery.service.settings")
    def test_init_scanners_none(self, mock_settings):
        mock_settings.discovery.ssdp_enabled = False
        mock_settings.discovery.mdns_enabled = False
        svc = DiscoveryService()
        assert len(svc._scanners) == 0

    @patch("atlas_brain.discovery.service.settings")
    def test_scan_result_to_device(self, mock_settings):
        mock_settings.discovery.ssdp_enabled = False
        mock_settings.discovery.mdns_enabled = False
        svc = DiscoveryService()

        scan_result = ScanResult(
            host="192.168.1.10",
            protocol="ssdp",
            device_type="roku",
            name="Living Room Roku",
            manufacturer="Roku Inc",
            model="Ultra 4800",
            port=8060,
            headers={"SERVER": "Roku/9.4"},
            services=["roku:ecp"],
        )

        device = svc._scan_result_to_device(scan_result)

        assert isinstance(device, DiscoveredDevice)
        assert device.device_id == "roku.192_168_1_10"
        assert device.name == "Living Room Roku"
        assert device.device_type == "roku"
        assert device.protocol == "ssdp"
        assert device.host == "192.168.1.10"
        assert device.port == 8060
        assert device.is_active is True
        assert device.auto_registered is False
        assert isinstance(device.id, UUID)
        assert isinstance(device.discovered_at, datetime)
        assert isinstance(device.last_seen_at, datetime)
        assert device.metadata["manufacturer"] == "Roku Inc"
        assert device.metadata["model"] == "Ultra 4800"
        assert device.metadata["headers"] == {"SERVER": "Roku/9.4"}
        assert device.metadata["services"] == ["roku:ecp"]

    @patch("atlas_brain.discovery.service.settings")
    def test_scan_result_to_device_defaults(self, mock_settings):
        """When ScanResult lacks name/device_type, _scan_result_to_device fills defaults."""
        mock_settings.discovery.ssdp_enabled = False
        mock_settings.discovery.mdns_enabled = False
        svc = DiscoveryService()

        scan_result = ScanResult(host="10.0.0.5", protocol="mdns")
        device = svc._scan_result_to_device(scan_result)

        assert device.device_id == "mdns.10_0_0_5"
        assert device.name == "Unknown (10.0.0.5)"
        assert device.device_type == "unknown"

    @pytest.mark.asyncio
    @patch("atlas_brain.discovery.service.settings")
    async def test_scan_with_mock_scanner(self, mock_settings):
        mock_settings.discovery.ssdp_enabled = False
        mock_settings.discovery.mdns_enabled = False
        mock_settings.discovery.persist_devices = False
        mock_settings.discovery.auto_register = False
        mock_settings.security.asset_tracking_enabled = False

        svc = DiscoveryService()

        # Build a fake scanner that returns a known result
        mock_scanner = AsyncMock(spec=BaseScanner)
        mock_scanner.protocol_name = "fake"
        mock_scanner.is_available = AsyncMock(return_value=True)
        mock_scanner.scan = AsyncMock(return_value=[
            ScanResult(
                host="192.168.1.77",
                protocol="fake",
                device_type="speaker",
                name="Test Speaker",
            ),
        ])

        svc._scanners = [mock_scanner]

        with patch("atlas_brain.discovery.service.get_device_repo"):
            devices = await svc.scan(timeout=2.0)

        assert len(devices) == 1
        assert devices[0].device_id == "speaker.192_168_1_77"
        assert devices[0].name == "Test Speaker"
        assert devices[0].device_type == "speaker"
        mock_scanner.scan.assert_awaited_once_with(timeout=2.0)

    @pytest.mark.asyncio
    @patch("atlas_brain.discovery.service.settings")
    async def test_scan_no_scanners(self, mock_settings):
        mock_settings.discovery.ssdp_enabled = False
        mock_settings.discovery.mdns_enabled = False
        svc = DiscoveryService()
        assert svc._scanners == []
        result = await svc.scan()
        assert result == []

    def test_device_type_to_asset_type_mapping(self):
        # All entries map to one of the expected asset types
        expected_asset_types = {"sensor", "drone", "vehicle"}
        for device_type, asset_type in DEVICE_TYPE_TO_ASSET_TYPE.items():
            assert asset_type in expected_asset_types, (
                "Unexpected asset type '%s' for device type '%s'" % (asset_type, device_type)
            )

        # Verify specific mappings mentioned in the source
        assert DEVICE_TYPE_TO_ASSET_TYPE["roku"] == "sensor"
        assert DEVICE_TYPE_TO_ASSET_TYPE["chromecast"] == "sensor"
        assert DEVICE_TYPE_TO_ASSET_TYPE["smart_tv"] == "sensor"
        assert DEVICE_TYPE_TO_ASSET_TYPE["media_renderer"] == "sensor"
        assert DEVICE_TYPE_TO_ASSET_TYPE["router"] == "sensor"
        assert DEVICE_TYPE_TO_ASSET_TYPE["speaker"] == "sensor"
        assert DEVICE_TYPE_TO_ASSET_TYPE["drone"] == "drone"
        assert DEVICE_TYPE_TO_ASSET_TYPE["vehicle"] == "vehicle"
        assert DEVICE_TYPE_TO_ASSET_TYPE["camera"] == "sensor"
        assert DEVICE_TYPE_TO_ASSET_TYPE["thermostat"] == "sensor"
        assert DEVICE_TYPE_TO_ASSET_TYPE["sensor"] == "sensor"

    @pytest.mark.asyncio
    @patch("atlas_brain.discovery.service.settings")
    async def test_get_discovered_devices_handles_error(self, mock_settings):
        mock_settings.discovery.ssdp_enabled = False
        mock_settings.discovery.mdns_enabled = False
        svc = DiscoveryService()

        with patch("atlas_brain.discovery.service.get_device_repo") as mock_repo:
            mock_repo.return_value.get_all_devices = AsyncMock(
                side_effect=RuntimeError("DB unavailable")
            )
            result = await svc.get_discovered_devices()

        assert result == []

    @pytest.mark.asyncio
    @patch("atlas_brain.discovery.service.settings")
    async def test_get_active_devices_handles_error(self, mock_settings):
        mock_settings.discovery.ssdp_enabled = False
        mock_settings.discovery.mdns_enabled = False
        svc = DiscoveryService()

        with patch("atlas_brain.discovery.service.get_device_repo") as mock_repo:
            mock_repo.return_value.get_active_devices = AsyncMock(
                side_effect=RuntimeError("DB unavailable")
            )
            result = await svc.get_active_devices()

        assert result == []

    @patch("atlas_brain.discovery.service.settings")
    def test_singleton_get_discovery_service(self, mock_settings):
        mock_settings.discovery.ssdp_enabled = False
        mock_settings.discovery.mdns_enabled = False

        # Reset global to ensure clean state
        import atlas_brain.discovery.service as svc_module
        svc_module._discovery_service = None

        svc1 = get_discovery_service()
        svc2 = get_discovery_service()
        assert svc1 is svc2

        # Cleanup: reset global so other tests are not affected
        svc_module._discovery_service = None

    @pytest.mark.asyncio
    @patch("atlas_brain.discovery.service.settings")
    async def test_scan_scanner_not_available(self, mock_settings):
        """Scanner that returns is_available=False is skipped."""
        mock_settings.discovery.ssdp_enabled = False
        mock_settings.discovery.mdns_enabled = False
        mock_settings.discovery.persist_devices = False
        mock_settings.discovery.auto_register = False
        mock_settings.security.asset_tracking_enabled = False

        svc = DiscoveryService()

        mock_scanner = AsyncMock(spec=BaseScanner)
        mock_scanner.protocol_name = "unavailable"
        mock_scanner.is_available = AsyncMock(return_value=False)
        mock_scanner.scan = AsyncMock(return_value=[])

        svc._scanners = [mock_scanner]

        with patch("atlas_brain.discovery.service.get_device_repo"):
            devices = await svc.scan(timeout=2.0)

        assert devices == []
        mock_scanner.scan.assert_not_awaited()

    @pytest.mark.asyncio
    @patch("atlas_brain.discovery.service.settings")
    async def test_scan_scanner_raises_exception(self, mock_settings):
        """Scanner that raises an exception does not crash the scan."""
        mock_settings.discovery.ssdp_enabled = False
        mock_settings.discovery.mdns_enabled = False
        mock_settings.discovery.persist_devices = False
        mock_settings.discovery.auto_register = False
        mock_settings.security.asset_tracking_enabled = False

        svc = DiscoveryService()

        failing_scanner = AsyncMock(spec=BaseScanner)
        failing_scanner.protocol_name = "broken"
        failing_scanner.is_available = AsyncMock(return_value=True)
        failing_scanner.scan = AsyncMock(side_effect=OSError("Network unreachable"))

        svc._scanners = [failing_scanner]

        with patch("atlas_brain.discovery.service.get_device_repo"):
            devices = await svc.scan(timeout=2.0)

        # Should return empty list without raising
        assert devices == []


# ---------------------------------------------------------------------------
# TestSSDPConstants
# ---------------------------------------------------------------------------


class TestSSDPConstants:
    """Verify SSDP module-level constants are correct."""

    def test_ssdp_multicast_address(self):
        assert SSDP_ADDR == "239.255.255.250"

    def test_ssdp_port(self):
        assert SSDP_PORT == 1900

    def test_device_patterns_keys(self):
        expected_types = {"roku", "chromecast", "smart_tv", "media_renderer", "router", "speaker"}
        assert set(DEVICE_PATTERNS.keys()) == expected_types

    def test_device_patterns_are_compiled_regex(self):
        import re
        for device_type, patterns in DEVICE_PATTERNS.items():
            assert len(patterns) > 0, "No patterns for device type '%s'" % device_type
            for pat in patterns:
                assert isinstance(pat, re.Pattern), (
                    "Pattern for '%s' is not a compiled regex" % device_type
                )
