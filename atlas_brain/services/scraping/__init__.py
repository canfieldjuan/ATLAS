"""
B2B review scraping with anti-detection.

Provides a curl_cffi-based HTTP client with TLS fingerprint spoofing,
proxy rotation, browser profile consistency, and per-domain rate limiting.
"""

from .capabilities import SourceCapabilityProfile, get_all_capabilities, get_capability
from .captcha import CaptchaSolution, CaptchaSolver, CaptchaType, detect_captcha, get_captcha_proxy, get_captcha_solver
from .client import AntiDetectionClient, get_scrape_client
from .proxy import ProxyConfig, ProxyManager
from .profiles import BrowserProfile, BrowserProfileManager
from .rate_limiter import DomainRateLimiter
from .sources import ReviewSource

__all__ = [
    "AntiDetectionClient",
    "get_scrape_client",
    "CaptchaSolution",
    "CaptchaSolver",
    "CaptchaType",
    "detect_captcha",
    "get_captcha_proxy",
    "get_captcha_solver",
    "ProxyConfig",
    "ProxyManager",
    "BrowserProfile",
    "BrowserProfileManager",
    "DomainRateLimiter",
    "ReviewSource",
    "SourceCapabilityProfile",
    "get_capability",
    "get_all_capabilities",
]
