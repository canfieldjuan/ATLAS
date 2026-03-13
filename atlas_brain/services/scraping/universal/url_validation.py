"""
URL validation for the universal scraper.

Blocks SSRF vectors: private IPs, localhost, metadata endpoints,
non-http(s) schemes, and internal hostnames.
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
    "metadata.google.internal",
    "metadata.internal",
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
    """Validate a URL for safety. Returns the normalized URL.

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

    # 5. Resolve and check IP addresses
    try:
        ip_str = hostname if _is_ip_literal(hostname) else _resolve_hostname(hostname)
        if ip_str:
            _check_ip_safety(ip_str, hostname_lower)
    except UnsafeURLError:
        raise
    except Exception as exc:
        # DNS resolution failure — allow the request to proceed;
        # the HTTP client will fail with a more descriptive error.
        logger.debug("DNS resolution failed for %s: %s", hostname, exc)

    return url


def validate_urls(urls: list[str]) -> list[str]:
    """Validate a list of URLs. Raises on the first unsafe URL."""
    return [validate_url(u) for u in urls]


def _is_ip_literal(hostname: str) -> bool:
    """Check if hostname is an IP address literal."""
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def _resolve_hostname(hostname: str) -> str | None:
    """Resolve hostname to IP for safety checks. Returns first IP or None."""
    try:
        results = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        if results:
            return results[0][4][0]
    except socket.gaierror:
        pass
    return None


def _check_ip_safety(ip_str: str, original_hostname: str) -> None:
    """Raise UnsafeURLError if the IP is private, loopback, link-local, etc."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return  # Not a valid IP — let HTTP client handle it

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
    # Explicit metadata IP check (covers 169.254.169.254)
    if ip_str in _METADATA_IPS:
        raise UnsafeURLError(
            f"Blocked cloud metadata endpoint: {ip_str}"
        )
    if addr.is_multicast:
        raise UnsafeURLError(
            f"Blocked multicast address: {ip_str} (hostname: {original_hostname})"
        )


def validate_redirect_url(redirect_url: str, original_url: str) -> str:
    """Validate a redirect target URL.

    Same rules as validate_url, but logs the redirect chain for auditing.
    Raises ``UnsafeURLError`` if the redirect target is unsafe.
    """
    logger.info(
        "Redirect: %s -> %s — validating target", original_url, redirect_url
    )
    return validate_url(redirect_url)
