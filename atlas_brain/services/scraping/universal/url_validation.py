"""
URL validation for the universal scraper.

Blocks SSRF vectors: private IPs, localhost, metadata endpoints,
non-http(s) schemes, and internal hostnames.

Checks ALL resolved IPs for a hostname (not just the first) and
fails closed when DNS resolution is suspicious.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

logger = logging.getLogger("atlas.services.scraping.universal.url_validation")

# Cloud metadata endpoints (AWS, GCP, Azure, DigitalOcean)
_METADATA_IPS = frozenset({
    "169.254.169.254",
})

_BLOCKED_HOSTNAMES = frozenset({
    "localhost",
    "localhost.localdomain",
    "metadata.google.internal",
    "metadata.internal",
})


class UnsafeURLError(ValueError):
    """Raised when a URL fails safety validation."""


def validate_url(url: str) -> str:
    """Validate a URL for safety. Returns the URL unchanged.

    Raises ``UnsafeURLError`` if the URL is blocked.
    """
    parsed = urlparse(url)

    # 1. Scheme must be http or https
    if parsed.scheme not in ("http", "https"):
        raise UnsafeURLError(
            f"Blocked URL scheme '{parsed.scheme}' — only http/https allowed"
        )

    # 2. Must have a hostname
    hostname = parsed.hostname
    if not hostname:
        raise UnsafeURLError("URL has no hostname")

    hostname_lower = hostname.lower().strip(".")

    # 3. Block known dangerous hostnames
    if hostname_lower in _BLOCKED_HOSTNAMES:
        raise UnsafeURLError(f"Blocked hostname: {hostname_lower}")

    # 4. Block .local and .internal TLDs
    if hostname_lower.endswith(".local") or hostname_lower.endswith(".internal"):
        raise UnsafeURLError(f"Blocked internal hostname: {hostname_lower}")

    # 5. Resolve and check ALL IP addresses (not just the first)
    if _is_ip_literal(hostname):
        _check_ip_safety(hostname, hostname_lower)
    else:
        resolved_ips = _resolve_all_ips(hostname)
        if not resolved_ips:
            # Fail closed: if we can't resolve, don't allow the request.
            # The HTTP client might resolve differently and hit an internal IP.
            raise UnsafeURLError(
                f"DNS resolution failed for {hostname} — blocking request "
                f"(fail-closed policy)"
            )
        for ip_str in resolved_ips:
            _check_ip_safety(ip_str, hostname_lower)

    return url


def validate_urls(urls: list[str]) -> list[str]:
    """Validate a list of URLs. Raises on the first unsafe URL."""
    return [validate_url(u) for u in urls]


def validate_redirect_url(redirect_url: str, original_url: str) -> str:
    """Validate a redirect target URL.

    Same rules as validate_url, but logs the redirect chain for auditing.
    Raises ``UnsafeURLError`` if the redirect target is unsafe.
    """
    logger.info(
        "Redirect: %s -> %s — validating target", original_url, redirect_url
    )
    return validate_url(redirect_url)


def _is_ip_literal(hostname: str) -> bool:
    """Check if hostname is an IP address literal."""
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def _resolve_all_ips(hostname: str) -> list[str]:
    """Resolve hostname to ALL IPs for safety checks.

    Returns a deduplicated list of IP strings, or empty list on failure.
    Checks both IPv4 and IPv6 records.
    """
    try:
        results = socket.getaddrinfo(
            hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
        seen: set[str] = set()
        ips: list[str] = []
        for family, _, _, _, sockaddr in results:
            ip = sockaddr[0]
            if ip not in seen:
                seen.add(ip)
                ips.append(ip)
        return ips
    except socket.gaierror:
        return []


def _check_ip_safety(ip_str: str, original_hostname: str) -> None:
    """Raise UnsafeURLError if the IP is private, loopback, link-local, etc."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        # Unparseable IP — fail closed
        raise UnsafeURLError(
            f"Unparseable IP address: {ip_str} (hostname: {original_hostname})"
        )

    if addr.is_loopback:
        raise UnsafeURLError(
            f"Blocked loopback address: {ip_str} (hostname: {original_hostname})"
        )
    if addr.is_private:
        raise UnsafeURLError(
            f"Blocked private IP: {ip_str} (hostname: {original_hostname})"
        )
    if addr.is_link_local:
        raise UnsafeURLError(
            f"Blocked link-local address: {ip_str} (hostname: {original_hostname})"
        )
    if addr.is_reserved:
        raise UnsafeURLError(
            f"Blocked reserved address: {ip_str} (hostname: {original_hostname})"
        )
    if addr.is_multicast:
        raise UnsafeURLError(
            f"Blocked multicast address: {ip_str} (hostname: {original_hostname})"
        )
    # Explicit metadata IP check (covers 169.254.169.254)
    if ip_str in _METADATA_IPS:
        raise UnsafeURLError(
            f"Blocked cloud metadata endpoint: {ip_str}"
        )
