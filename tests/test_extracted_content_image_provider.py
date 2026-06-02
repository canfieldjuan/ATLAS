from __future__ import annotations

import json
from typing import Any
import urllib.request
from urllib.parse import parse_qs, urlparse

import pytest

import extracted_content_pipeline.content_image_provider as image_provider_mod
from extracted_content_pipeline.content_image_provider import (
    ContentImageRequest,
    UnsplashContentImageProvider,
    UnsplashContentImageProviderConfig,
)


class _Response:
    def __init__(self, payload: Any, status: int = 200) -> None:
        self.payload = payload
        self.status = status

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def getcode(self) -> int:
        return self.status


class _Transport:
    def __init__(self, *responses: _Response) -> None:
        self.responses = list(responses)
        self.requests: list[urllib.request.Request] = []

    def __call__(self, request: urllib.request.Request, timeout: float) -> _Response:
        assert timeout == 7.0
        self.requests.append(request)
        assert self.responses, f"unexpected request: {request.full_url}"
        return self.responses.pop(0)


def _provider(transport: _Transport, *, base_url: str = "https://api.unsplash.test"):
    return UnsplashContentImageProvider(
        UnsplashContentImageProviderConfig(
            enabled=True,
            access_key="unsplash-key",
            base_url=base_url,
            timeout_seconds=7.0,
        ),
        opener=transport,
    )


@pytest.mark.asyncio
async def test_unsplash_provider_selects_first_landscape_and_tracks_download() -> None:
    transport = _Transport(
        _Response({
            "results": [{
                "id": "photo-1",
                "alt_description": "Support team reviewing customer questions",
                "urls": {"regular": "https://images.unsplash.test/photo-1.jpg"},
                "links": {
                    "html": "https://unsplash.test/photos/photo-1",
                    "download_location": "https://api.unsplash.test/photos/photo-1/download",
                },
                "user": {
                    "name": "Ada Lens",
                    "links": {"html": "https://unsplash.test/@ada"},
                },
            }],
        }),
        _Response({"url": "https://images.unsplash.test/photo-1.jpg"}),
    )
    provider = _provider(transport)

    asset = await provider.select_image(
        ContentImageRequest(
            asset_type="landing_page",
            slot="hero",
            title="Support FAQ gaps",
            query_terms=("SaaS help desk", "customer support"),
        )
    )

    assert asset is not None
    assert asset.url == "https://images.unsplash.test/photo-1.jpg"
    assert asset.provider == "unsplash"
    assert asset.attribution_name == "Ada Lens"
    assert asset.attribution_url == "https://unsplash.test/@ada"
    assert asset.source_id == "photo-1"
    assert [request.get_method() for request in transport.requests] == ["GET", "GET"]
    assert [
        request.get_header("Authorization") for request in transport.requests
    ] == ["Client-ID unsplash-key", "Client-ID unsplash-key"]

    search_url = urlparse(transport.requests[0].full_url)
    assert search_url.path == "/search/photos"
    params = parse_qs(search_url.query)
    assert params["query"] == ["Support FAQ gaps SaaS help desk customer support"]
    assert params["per_page"] == ["1"]
    assert params["orientation"] == ["landscape"]
    assert transport.requests[1].full_url == (
        "https://api.unsplash.test/photos/photo-1/download"
    )


@pytest.mark.asyncio
async def test_unsplash_provider_offloads_blocking_transport_to_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _Transport(_Response({"results": []}))
    provider = _provider(transport)
    calls = []

    async def fake_to_thread(func, *args):
        calls.append((func, args))
        return func(*args)

    monkeypatch.setattr(image_provider_mod.asyncio, "to_thread", fake_to_thread)

    asset = await provider.select_image(
        ContentImageRequest(asset_type="landing_page", slot="hero", title="FAQ gaps")
    )

    assert asset is None
    assert len(calls) == 1
    assert calls[0][0].__name__ == "_select_image_sync"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    (
        {"results": []},
        {"results": [{"urls": {}, "links": {}}]},
        {"results": [{"urls": {"regular": "http://images.test/photo.jpg"}, "links": {}}]},
    ),
)
async def test_unsplash_provider_fails_closed_on_malformed_envelopes(
    payload: dict[str, Any],
) -> None:
    transport = _Transport(_Response(payload))
    provider = _provider(transport)

    asset = await provider.select_image(
        ContentImageRequest(asset_type="blog_post", slot="cover", title="Pricing pressure")
    )

    assert asset is None


@pytest.mark.asyncio
async def test_unsplash_provider_requires_download_tracking_success() -> None:
    transport = _Transport(
        _Response({
            "results": [{
                "id": "photo-1",
                "urls": {"regular": "https://images.unsplash.test/photo-1.jpg"},
                "links": {
                    "download_location": "https://api.unsplash.test/photos/photo-1/download"
                },
            }],
        }),
        _Response({"error": "rate limited"}, status=429),
    )
    provider = _provider(transport)

    asset = await provider.select_image(
        ContentImageRequest(asset_type="landing_page", slot="hero", title="FAQ gaps")
    )

    assert asset is None
    assert len(transport.requests) == 2


@pytest.mark.asyncio
async def test_unsplash_provider_rejects_unsafe_base_url_without_transport_call() -> None:
    transport = _Transport()
    provider = _provider(transport, base_url="http://localhost:9000")

    asset = await provider.select_image(
        ContentImageRequest(asset_type="landing_page", slot="hero", title="FAQ gaps")
    )

    assert asset is None
    assert transport.requests == []
