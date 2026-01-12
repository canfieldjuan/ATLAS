"""
SSDP (Simple Service Discovery Protocol) scanner.

Discovers UPnP devices on the local network using multicast M-SEARCH requests.
Commonly finds: Roku, smart TVs, media renderers, routers, etc.
"""

import asyncio
import logging
import re
import socket
from typing import Optional
from urllib.parse import urlparse

from .base import BaseScanner, ScanResult

logger = logging.getLogger("atlas.discovery.scanners.ssdp")

# SSDP multicast address and port
SSDP_ADDR = "239.255.255.250"
SSDP_PORT = 1900

# M-SEARCH request template
MSEARCH_TEMPLATE = (
    "M-SEARCH * HTTP/1.1\r\n"
    "HOST: {addr}:{port}\r\n"
    "MAN: \"ssdp:discover\"\r\n"
    "MX: {mx}\r\n"
    "ST: {st}\r\n"
    "\r\n"
)

# Search targets for different device types
SEARCH_TARGETS = [
    "ssdp:all",  # All devices
    "upnp:rootdevice",  # Root devices
    "roku:ecp",  # Roku devices specifically
    "urn:dial-multiscreen-org:service:dial:1",  # DIAL (Chromecast, smart TVs)
]

# Device type identification patterns
DEVICE_PATTERNS = {
    "roku": [
        re.compile(r"roku", re.IGNORECASE),
        re.compile(r"roku:ecp", re.IGNORECASE),
    ],
    "chromecast": [
        re.compile(r"chromecast", re.IGNORECASE),
        re.compile(r"google.*cast", re.IGNORECASE),
    ],
    "smart_tv": [
        re.compile(r"samsung.*tv", re.IGNORECASE),
        re.compile(r"lg.*tv", re.IGNORECASE),
        re.compile(r"sony.*bravia", re.IGNORECASE),
        re.compile(r"philips.*tv", re.IGNORECASE),
        re.compile(r"urn:schemas-upnp-org:device:MediaRenderer", re.IGNORECASE),
    ],
    "media_renderer": [
        re.compile(r"MediaRenderer", re.IGNORECASE),
    ],
    "router": [
        re.compile(r"InternetGatewayDevice", re.IGNORECASE),
        re.compile(r"WANDevice", re.IGNORECASE),
    ],
    "speaker": [
        re.compile(r"sonos", re.IGNORECASE),
        re.compile(r"speaker", re.IGNORECASE),
    ],
}


class SSDPScanner(BaseScanner):
    """
    SSDP/UPnP network scanner.

    Sends M-SEARCH multicast requests and collects responses
    to discover devices on the local network.
    """

    @property
    def protocol_name(self) -> str:
        return "ssdp"

    async def scan(self, timeout: float = 5.0) -> list[ScanResult]:
        """
        Scan the network for SSDP devices.

        Args:
            timeout: How long to wait for responses (seconds)

        Returns:
            List of discovered devices
        """
        logger.info("Starting SSDP scan (timeout=%.1fs)", timeout)

        # Collect responses from all search targets
        all_responses: dict[str, dict] = {}  # keyed by host to deduplicate

        for st in SEARCH_TARGETS:
            try:
                responses = await self._send_msearch(st, timeout=timeout / len(SEARCH_TARGETS))
                for host, data in responses.items():
                    if host in all_responses:
                        # Merge data from multiple responses
                        all_responses[host]["headers"].update(data.get("headers", {}))
                        all_responses[host]["services"].extend(data.get("services", []))
                    else:
                        all_responses[host] = data
            except Exception as e:
                logger.warning("SSDP search for %s failed: %s", st, e)

        # Convert to ScanResults
        results = []
        for host, data in all_responses.items():
            result = ScanResult(
                host=host,
                protocol="ssdp",
                port=data.get("port"),
                name=data.get("name"),
                manufacturer=data.get("manufacturer"),
                model=data.get("model"),
                headers=data.get("headers", {}),
                services=list(set(data.get("services", []))),
                raw_data=data,
            )

            # Identify device type
            result.device_type = self.identify_device_type(result)
            results.append(result)

        logger.info("SSDP scan complete: found %d devices", len(results))
        return results

    async def _send_msearch(
        self,
        search_target: str,
        timeout: float = 3.0,
    ) -> dict[str, dict]:
        """
        Send M-SEARCH request and collect responses.

        Args:
            search_target: The ST (search target) value
            timeout: Response collection timeout

        Returns:
            Dict mapping host -> response data
        """
        # Build the M-SEARCH request
        request = MSEARCH_TEMPLATE.format(
            addr=SSDP_ADDR,
            port=SSDP_PORT,
            mx=int(timeout),
            st=search_target,
        ).encode()

        responses: dict[str, dict] = {}

        # Create UDP socket
        loop = asyncio.get_event_loop()

        try:
            # Create socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setblocking(False)

            # Bind to any available port
            sock.bind(("", 0))

            # Send M-SEARCH request
            sock.sendto(request, (SSDP_ADDR, SSDP_PORT))
            logger.debug("Sent M-SEARCH for %s", search_target)

            # Collect responses
            end_time = loop.time() + timeout
            while loop.time() < end_time:
                try:
                    remaining = end_time - loop.time()
                    if remaining <= 0:
                        break

                    # Wait for response with timeout
                    data, addr = await asyncio.wait_for(
                        loop.sock_recvfrom(sock, 4096),
                        timeout=remaining,
                    )

                    host = addr[0]
                    response_data = self._parse_response(data.decode("utf-8", errors="ignore"))

                    if host not in responses:
                        responses[host] = {
                            "headers": {},
                            "services": [],
                            "port": None,
                            "name": None,
                            "manufacturer": None,
                            "model": None,
                        }

                    # Merge response data
                    responses[host]["headers"].update(response_data.get("headers", {}))
                    if response_data.get("st"):
                        responses[host]["services"].append(response_data["st"])
                    if response_data.get("location"):
                        # Extract port from location URL
                        try:
                            parsed = urlparse(response_data["location"])
                            responses[host]["port"] = parsed.port
                        except Exception:
                            pass

                    # Try to get friendly name from headers
                    headers = responses[host]["headers"]
                    if not responses[host]["name"]:
                        responses[host]["name"] = (
                            headers.get("X-FRIENDLY-NAME")
                            or headers.get("SERVER")
                            or headers.get("USN", "").split("::")[-1]
                        )

                    logger.debug("SSDP response from %s: %s", host, response_data.get("st", "unknown"))

                except asyncio.TimeoutError:
                    break
                except Exception as e:
                    logger.debug("Error receiving SSDP response: %s", e)
                    break

        except Exception as e:
            logger.error("SSDP socket error: %s", e)
        finally:
            try:
                sock.close()
            except Exception:
                pass

        return responses

    def _parse_response(self, response: str) -> dict:
        """Parse an SSDP response into a dictionary."""
        result = {"headers": {}}

        lines = response.split("\r\n")

        for line in lines[1:]:  # Skip status line
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().upper()
                value = value.strip()
                result["headers"][key] = value

                # Extract specific fields
                if key == "ST":
                    result["st"] = value
                elif key == "LOCATION":
                    result["location"] = value
                elif key == "SERVER":
                    result["server"] = value
                elif key == "USN":
                    result["usn"] = value

        return result

    def identify_device_type(self, result: ScanResult) -> Optional[str]:
        """
        Identify the device type from SSDP response data.

        Checks headers, services, and names against known patterns.
        """
        # Build a string of all searchable data
        search_text = " ".join([
            result.name or "",
            result.manufacturer or "",
            result.model or "",
            " ".join(result.services),
            " ".join(f"{k}={v}" for k, v in result.headers.items()),
        ])

        # Check against patterns
        for device_type, patterns in DEVICE_PATTERNS.items():
            for pattern in patterns:
                if pattern.search(search_text):
                    logger.debug(
                        "Identified %s as %s (matched: %s)",
                        result.host, device_type, pattern.pattern
                    )
                    return device_type

        # Default to generic UPnP device
        return "upnp_device"
