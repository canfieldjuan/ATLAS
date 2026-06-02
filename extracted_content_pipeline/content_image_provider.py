"""Content image provider ports and the Unsplash adapter."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import ipaddress
import json
from typing import Any, Callable, Protocol
from urllib.parse import urlencode, urljoin, urlparse
import urllib.request


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class ContentImageRequest:
    """Trusted server-side context used to pick an image for a content asset."""

    asset_type: str
    slot: str
    title: str = ""
    query_terms: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def query(self) -> str:
        parts = [self.title, *self.query_terms]
        return " ".join(str(part).strip() for part in parts if str(part).strip())


@dataclass(frozen=True)
class ContentImageAsset:
    """Selected image payload stored on generated drafts."""

    url: str
    provider: str
    alt_text: str = ""
    attribution_name: str = ""
    attribution_url: str = ""
    source_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> JsonDict:
        return {
            "url": self.url,
            "provider": self.provider,
            "alt_text": self.alt_text,
            "attribution_name": self.attribution_name,
            "attribution_url": self.attribution_url,
            "source_id": self.source_id,
            "metadata": dict(self.metadata),
        }


class ContentImageProvider(Protocol):
    async def select_image(
        self,
        request: ContentImageRequest,
    ) -> ContentImageAsset | None:
        """Return one image for a content asset, or None when unavailable."""


@dataclass(frozen=True)
class UnsplashContentImageProviderConfig:
    enabled: bool = False
    access_key: str = ""
    base_url: str = "https://api.unsplash.com"
    timeout_seconds: float = 10.0


Opener = Callable[[urllib.request.Request, float], Any]


class UnsplashContentImageProvider:
    """Unsplash search adapter with required download tracking."""

    def __init__(
        self,
        config: UnsplashContentImageProviderConfig,
        *,
        opener: Opener | None = None,
    ) -> None:
        self._config = config
        self._opener = opener or _urlopen

    async def select_image(
        self,
        request: ContentImageRequest,
    ) -> ContentImageAsset | None:
        return await asyncio.to_thread(self._select_image_sync, request)

    def _select_image_sync(
        self,
        request: ContentImageRequest,
    ) -> ContentImageAsset | None:
        if not self._config.enabled or not self._config.access_key.strip():
            return None
        base_url = _safe_https_base_url(self._config.base_url)
        if not base_url:
            return None
        query = request.query()
        if not query:
            return None
        search_url = urljoin(base_url + "/", "search/photos")
        params = {
            'query': query,
            'per_page': 1,
            'orientation': 'landscape',
            'content_filter': 'high',
        }
        search_url = f"{search_url}?{urlencode(params)}"
        try:
            payload = self._fetch_json(search_url)
            asset, download_url = _asset_from_unsplash_payload(payload)
            if asset is None or not _safe_https_url(download_url):
                return None
            if not self._track_download(download_url):
                return None
            return asset
        except Exception:
            return None

    def _fetch_json(self, url: str) -> Mapping[str, Any]:
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Client-ID {self._config.access_key.strip()}",
                "Accept": "application/json",
                "User-Agent": "Atlas-Content-Ops/1.0",
            },
        )
        with self._opener(request, _timeout(self._config.timeout_seconds)) as response:
            status = int(getattr(response, "status", 0) or response.getcode())
            if status < 200 or status >= 300:
                return {}
            decoded = json.loads(response.read().decode("utf-8"))
            return decoded if isinstance(decoded, Mapping) else {}

    def _track_download(self, url: str) -> bool:
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Client-ID {self._config.access_key.strip()}",
                "Accept": "application/json",
                "User-Agent": "Atlas-Content-Ops/1.0",
            },
        )
        with self._opener(request, _timeout(self._config.timeout_seconds)) as response:
            status = int(getattr(response, "status", 0) or response.getcode())
            return 200 <= status < 300


def _asset_from_unsplash_payload(
    payload: Mapping[str, Any],
) -> tuple[ContentImageAsset | None, str]:
    results = payload.get("results")
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
        return None, ""
    first = next((item for item in results if isinstance(item, Mapping)), None)
    if first is None:
        return None, ""
    urls = first.get("urls") if isinstance(first.get("urls"), Mapping) else {}
    image_url = _clean_text(urls.get("regular") or urls.get("full") or urls.get("small"))
    links = first.get("links") if isinstance(first.get("links"), Mapping) else {}
    download_url = _clean_text(links.get("download_location"))
    if not image_url or not _safe_https_url(image_url):
        return None, ""
    user = first.get("user") if isinstance(first.get("user"), Mapping) else {}
    user_links = user.get("links") if isinstance(user.get("links"), Mapping) else {}
    asset = ContentImageAsset(
        url=image_url,
        provider="unsplash",
        alt_text=_clean_text(first.get("alt_description") or first.get("description")),
        attribution_name=_clean_text(user.get("name")),
        attribution_url=_clean_text(user_links.get("html")),
        source_id=_clean_text(first.get("id")),
        metadata={"source": "unsplash"},
    )
    return asset, download_url


def _urlopen(request: urllib.request.Request, timeout: float) -> Any:
    return urllib.request.urlopen(request, timeout=timeout)


def _timeout(value: float) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return 10.0
    return timeout if timeout > 0 else 10.0


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_https_base_url(value: Any) -> str:
    parsed = urlparse(_clean_text(value))
    if parsed.path not in ("", "/"):
        return ""
    if not _safe_parsed_https_url(parsed):
        return ""
    return f"https://{parsed.netloc}"


def _safe_https_url(value: Any) -> bool:
    return _safe_parsed_https_url(urlparse(_clean_text(value)))


def _safe_parsed_https_url(parsed: Any) -> bool:
    if parsed.scheme != "https" or not parsed.netloc:
        return False
    host = parsed.hostname or ""
    if host.lower() == "localhost" or host.endswith(".local"):
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return True
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
    )


__all__ = [
    "ContentImageAsset",
    "ContentImageProvider",
    "ContentImageRequest",
    "UnsplashContentImageProvider",
    "UnsplashContentImageProviderConfig",
]
