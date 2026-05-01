from __future__ import annotations

import json
import logging
import re
from typing import Any
from urllib.parse import parse_qs, parse_qsl, urlencode, urlparse, urlunsplit, urlsplit

from bs4 import BeautifulSoup

logger = logging.getLogger("atlas.autonomous.tasks.google_news")

_GOOGLE_NEWS_HOST = "news.google.com"
_GOOGLE_NEWS_PATH_PREFIX = "/rss/articles/"
_GOOGLE_NEWS_RPC_ID = "Fbv4je"
_GOOGLE_NEWS_DECODE_URL = (
    "https://news.google.com/_/DotsSplashUi/data/batchexecute"
    "?rpcids=Fbv4je"
)
_GOOGLE_NEWS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AtlasBot/1.0)",
    "Accept": "text/html,application/xhtml+xml",
}
_GOOGLE_NEWS_FORM_HEADERS = {
    **_GOOGLE_NEWS_HEADERS,
    "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
    "Referer": "https://news.google.com/",
}
_GOOGLE_NEWS_FEATURE_FLAGS = ["FINANCE_TOP_INDICES", "WEB_TEST_1_0_0"]
_GOOGLE_NEWS_REQUEST_BUILD = "655000234"
_GARTURLRES_PATTERN = re.compile(r'\["garturlres","([^"]+)",\d+\]')


def is_google_news_wrapper_url(url: str) -> bool:
    """Return True when the URL points at a Google News RSS article wrapper."""
    parsed = urlparse(url)
    return (
        parsed.scheme in {"http", "https"}
        and parsed.netloc == _GOOGLE_NEWS_HOST
        and parsed.path.startswith(_GOOGLE_NEWS_PATH_PREFIX)
    )


def _article_locale_context(url: str) -> tuple[str, str, str]:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    locale = query.get("hl", ["en-US"])[0]
    region = query.get("gl", ["US"])[0]
    ceid = query.get("ceid", [f"{region}:{locale.split('-', 1)[0]}"])[0]
    return locale, region, ceid


def _extract_decode_inputs(html: str) -> tuple[str, int, str] | None:
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.find(
        attrs={
            "data-n-a-id": True,
            "data-n-a-ts": True,
            "data-n-a-sg": True,
        }
    )
    if not tag:
        return None
    try:
        article_id = str(tag.get("data-n-a-id", "")).strip()
        timestamp = int(str(tag.get("data-n-a-ts", "")).strip())
        signature = str(tag.get("data-n-a-sg", "")).strip()
    except (TypeError, ValueError):
        return None
    if not article_id or not signature:
        return None
    return article_id, timestamp, signature


def _build_decode_payload(
    article_id: str,
    timestamp: int,
    signature: str,
    locale: str,
    region: str,
    ceid: str,
) -> str:
    request = [
        "garturlreq",
        [
            [
                locale,
                region,
                _GOOGLE_NEWS_FEATURE_FLAGS,
                None,
                None,
                1,
                1,
                ceid,
                None,
                180,
                None,
                None,
                None,
                None,
                None,
                0,
                None,
                None,
                [1608992183, 723341000],
            ],
            locale,
            region,
            1,
            [2, 3, 4, 8],
            1,
            0,
            _GOOGLE_NEWS_REQUEST_BUILD,
            0,
            0,
            None,
            0,
        ],
        article_id,
        timestamp,
        signature,
    ]
    return json.dumps([[[_GOOGLE_NEWS_RPC_ID, json.dumps(request), None, "generic"]]])


def _extract_resolved_url(payload_text: str) -> str | None:
    for candidate in (payload_text, payload_text.replace('\\"', '"')):
        match = _GARTURLRES_PATTERN.search(candidate)
        if not match:
            continue
        try:
            resolved = json.loads(f'"{match.group(1)}"')
        except json.JSONDecodeError:
            resolved = match.group(1).replace("\\/", "/").replace("\\u003d", "=")
        if resolved.startswith(("http://", "https://")):
            return _normalize_resolved_url(resolved)
    return None


def _normalize_resolved_url(url: str) -> str:
    cleaned = (
        url.replace("\\u003d", "=")
        .replace("\\u0026", "&")
        .replace("\\u002F", "/")
        .replace("\\/", "/")
    )
    parsed = urlsplit(cleaned)
    query = urlencode(
        [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if not k.startswith("gaa_")],
        doseq=True,
    )
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, ""))


async def resolve_google_news_url(
    wrapper_url: str,
    timeout: float = 20.0,
    client: Any | None = None,
) -> str | None:
    """Resolve a Google News wrapper URL to the publisher article URL."""
    if not is_google_news_wrapper_url(wrapper_url):
        return wrapper_url

    import httpx

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(follow_redirects=True, timeout=timeout)

    try:
        wrapper_response = await client.get(wrapper_url, headers=_GOOGLE_NEWS_HEADERS)
        wrapper_response.raise_for_status()

        inputs = _extract_decode_inputs(wrapper_response.text)
        if not inputs:
            return None

        locale, region, ceid = _article_locale_context(str(wrapper_response.url))
        payload = _build_decode_payload(*inputs, locale, region, ceid)
        decode_response = await client.post(
            _GOOGLE_NEWS_DECODE_URL,
            data={"f.req": payload},
            headers=_GOOGLE_NEWS_FORM_HEADERS,
        )
        decode_response.raise_for_status()
        return _extract_resolved_url(decode_response.text)
    except Exception:
        logger.debug("Failed to resolve Google News wrapper URL: %s", wrapper_url, exc_info=True)
        return None
    finally:
        if own_client:
            await client.aclose()
