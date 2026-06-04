"""Content Ops brand voice profile management routes."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from html.parser import HTMLParser
import http.client
import ipaddress
import re
import socket
import ssl
import urllib.error
import urllib.parse
import uuid as _uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from .._content_ops_brand_voice_profiles import (
    ContentOpsBrandVoiceProfileRecord,
    archive_brand_voice_profile,
    create_brand_voice_profile,
    list_brand_voice_profiles,
    update_brand_voice_profile,
)
from ..auth.dependencies import AuthUser


PoolProvider = Callable[[], Any | Awaitable[Any]]
AuthDependency = Callable[..., AuthUser | Awaitable[AuthUser]]
_SAMPLE_URL_FETCH_TIMEOUT_SECONDS = 10
_SAMPLE_URL_MAX_BYTES = 512_000
_SAMPLE_TEXT_MAX_CHARS = 24_000
_SAMPLE_URL_USER_AGENT = "Atlas-Content-Ops/1.0"
_HTML_CONTENT_TYPES = ("text/html", "application/xhtml+xml")


class UpsertBrandVoiceProfileRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=160)
    descriptors: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    exemplars: tuple[str, ...] = Field(default_factory=tuple, max_length=3)
    banned_terms: tuple[str, ...] = Field(default_factory=tuple, max_length=20)
    preferred_pov: str | None = Field(default=None, max_length=40)
    reading_level: str | None = Field(default=None, max_length=80)
    metadata: dict[str, object] = Field(default_factory=dict)


class BrandVoiceProfileView(BaseModel):
    id: str
    account_id: str
    name: str
    descriptors: list[str]
    exemplars: list[str]
    banned_terms: list[str]
    preferred_pov: str | None = None
    reading_level: str | None = None
    metadata: dict[str, object]
    created_at: str
    updated_at: str
    archived_at: str | None = None


class BrandVoiceSampleUrlRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2048)


class BrandVoiceSampleUrlView(BaseModel):
    url: str
    title: str | None = None
    text: str
    source_character_count: int


def _default_pool_provider() -> Any:
    from ..storage.database import get_db_pool

    return get_db_pool()


def create_content_ops_brand_voice_profiles_router(
    *,
    pool_provider: PoolProvider = _default_pool_provider,
    auth_dependency: AuthDependency,
) -> APIRouter:
    """Create tenant-scoped Content Ops brand voice profile routes."""

    router = APIRouter(
        prefix="/content-ops/brand-voice-profiles",
        tags=["content-ops"],
    )

    @router.get("", response_model=list[BrandVoiceProfileView])
    async def list_profiles(
        user: AuthUser = Depends(auth_dependency),
    ) -> list[BrandVoiceProfileView]:
        pool = await _resolve_ready_pool(pool_provider)
        account_id = _account_uuid(user)
        records = await list_brand_voice_profiles(pool, account_id=account_id)
        return [_record_to_view(record) for record in records]

    @router.post("", response_model=BrandVoiceProfileView, status_code=201)
    async def add_profile(
        body: UpsertBrandVoiceProfileRequest,
        user: AuthUser = Depends(auth_dependency),
    ) -> BrandVoiceProfileView:
        _require_profile_admin(user)
        pool = await _resolve_ready_pool(pool_provider)
        account_id = _account_uuid(user)
        try:
            record = await create_brand_voice_profile(
                pool,
                account_id=account_id,
                payload=body.model_dump(),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            if exc.__class__.__name__ in ("UniqueViolationError", "IntegrityError"):
                raise HTTPException(
                    status_code=409,
                    detail="A brand voice profile with that name already exists.",
                )
            raise
        return _record_to_view(record)

    @router.post("/sample-url", response_model=BrandVoiceSampleUrlView)
    async def sample_url(
        body: BrandVoiceSampleUrlRequest,
        user: AuthUser = Depends(auth_dependency),
    ) -> BrandVoiceSampleUrlView:
        _require_profile_admin(user)
        sample = await asyncio.to_thread(_extract_brand_voice_sample_from_url, body.url)
        return BrandVoiceSampleUrlView(
            url=sample.url,
            title=sample.title,
            text=sample.text,
            source_character_count=sample.source_character_count,
        )

    @router.put("/{profile_id}", response_model=BrandVoiceProfileView)
    async def update_profile(
        profile_id: str,
        body: UpsertBrandVoiceProfileRequest,
        user: AuthUser = Depends(auth_dependency),
    ) -> BrandVoiceProfileView:
        _require_profile_admin(user)
        pool = await _resolve_ready_pool(pool_provider)
        account_id = _account_uuid(user)
        parsed_profile_id = _profile_uuid(profile_id)
        try:
            record = await update_brand_voice_profile(
                pool,
                account_id=account_id,
                profile_id=parsed_profile_id,
                payload=body.model_dump(),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            if exc.__class__.__name__ in ("UniqueViolationError", "IntegrityError"):
                raise HTTPException(
                    status_code=409,
                    detail="A brand voice profile with that name already exists.",
                )
            raise
        if record is None:
            raise HTTPException(status_code=404, detail="Brand voice profile not found")
        return _record_to_view(record)

    @router.delete("/{profile_id}", status_code=204)
    async def delete_profile(
        profile_id: str,
        user: AuthUser = Depends(auth_dependency),
    ) -> None:
        _require_profile_admin(user)
        pool = await _resolve_ready_pool(pool_provider)
        account_id = _account_uuid(user)
        archived = await archive_brand_voice_profile(
            pool,
            account_id=account_id,
            profile_id=_profile_uuid(profile_id),
        )
        if not archived:
            raise HTTPException(status_code=404, detail="Brand voice profile not found")
        return None

    return router


async def _resolve_ready_pool(pool_provider: PoolProvider) -> Any:
    pool = pool_provider()
    if hasattr(pool, "__await__"):
        pool = await pool
    if pool is None or getattr(pool, "is_initialized", True) is False:
        raise HTTPException(status_code=503, detail="Database not ready")
    return pool


def _account_uuid(user: AuthUser) -> _uuid.UUID:
    try:
        return _uuid.UUID(str(user.account_id))
    except (TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid tenant scope")


def _profile_uuid(profile_id: str) -> _uuid.UUID:
    try:
        return _uuid.UUID(str(profile_id))
    except (TypeError, ValueError):
        raise HTTPException(status_code=404, detail="Brand voice profile not found")


def _require_profile_admin(user: AuthUser) -> None:
    role = str(getattr(user, "role", "") or "").strip().lower()
    if bool(getattr(user, "is_admin", False)) or role in {"owner", "admin"}:
        return
    raise HTTPException(status_code=403, detail="Admin access required")


def _record_to_view(record: ContentOpsBrandVoiceProfileRecord) -> BrandVoiceProfileView:
    return BrandVoiceProfileView(
        id=str(record.id),
        account_id=str(record.account_id),
        name=record.name,
        descriptors=list(record.descriptors),
        exemplars=list(record.exemplars),
        banned_terms=list(record.banned_terms),
        preferred_pov=record.preferred_pov,
        reading_level=record.reading_level,
        metadata=dict(record.metadata),
        created_at=_fmt_time(record.created_at) or "",
        updated_at=_fmt_time(record.updated_at) or "",
        archived_at=_fmt_time(record.archived_at),
    )


def _fmt_time(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


@dataclass(frozen=True)
class _SampleUrlFetchTarget:
    url: str
    host: str
    port: int
    path: str
    host_header: str
    connect_host: str


@dataclass(frozen=True)
class _ExtractedBrandVoiceSample:
    url: str
    title: str | None
    text: str
    source_character_count: int


class _ReadableHtmlTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._head_depth = 0
        self._title_depth = 0
        self._title_parts: list[str] = []
        self._text_parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        lowered = tag.lower()
        if lowered in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        elif lowered == "head":
            self._head_depth += 1
        elif lowered == "title":
            self._title_depth += 1

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style", "noscript", "svg"}:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif lowered == "head":
            self._head_depth = max(0, self._head_depth - 1)
        elif lowered == "title":
            self._title_depth = max(0, self._title_depth - 1)

    def handle_data(self, data: str) -> None:
        text = _normalize_readable_text(data)
        if not text:
            return
        if self._title_depth > 0:
            self._title_parts.append(text)
            return
        if self._skip_depth == 0 and self._head_depth == 0:
            self._text_parts.append(text)

    @property
    def title(self) -> str | None:
        return _normalize_readable_text(" ".join(self._title_parts)) or None

    @property
    def text(self) -> str:
        return _normalize_readable_text(" ".join(self._text_parts))


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(
        self,
        host: str,
        *,
        port: int,
        connect_host: str,
        timeout: int,
    ) -> None:
        super().__init__(
            host,
            port=port,
            timeout=timeout,
            context=ssl.create_default_context(),
        )
        self._connect_host = connect_host

    def connect(self) -> None:
        self.sock = self._create_connection(
            (self._connect_host, self.port),
            self.timeout,
            self.source_address,
        )
        if self._tunnel_host:
            self._tunnel()
        self.sock = self._context.wrap_socket(self.sock, server_hostname=self.host)


_PINNED_HTTPS_CONNECTION_CLASS = _PinnedHTTPSConnection


class _PinnedSampleUrlResponse:
    def __init__(
        self,
        response: http.client.HTTPResponse,
        connection: http.client.HTTPSConnection,
    ) -> None:
        self._response = response
        self._connection = connection
        self.status = response.status

    def read(self, size: int) -> bytes:
        return self._response.read(size)

    def getheader(self, name: str, default: str = "") -> str:
        return self._response.getheader(name, default)

    def close(self) -> None:
        self._response.close()
        self._connection.close()


def _extract_brand_voice_sample_from_url(url: str) -> _ExtractedBrandVoiceSample:
    body, content_type, resolved_url = _read_bounded_https_sample_url(
        url,
        max_bytes=_SAMPLE_URL_MAX_BYTES,
    )
    title, text = _extract_readable_text(body, content_type=content_type)
    if not text:
        raise HTTPException(
            status_code=400,
            detail="Sample URL did not include readable text.",
        )
    source_character_count = len(text)
    return _ExtractedBrandVoiceSample(
        url=resolved_url,
        title=title,
        text=text[:_SAMPLE_TEXT_MAX_CHARS],
        source_character_count=source_character_count,
    )


def _read_bounded_https_sample_url(
    value: str,
    *,
    max_bytes: int,
) -> tuple[bytes, str, str]:
    target = _validate_https_sample_url_fetch_target(value)
    response: Any | None = None
    try:
        response = _open_https_sample_url_request(
            target,
            timeout=_SAMPLE_URL_FETCH_TIMEOUT_SECONDS,
        )
        status = int(getattr(response, "status", 0) or 0)
        if 300 <= status < 400:
            raise HTTPException(
                status_code=400,
                detail="Sample URL redirects are not allowed.",
            )
        if status < 200 or status >= 300:
            raise HTTPException(
                status_code=400,
                detail="Sample URL could not be fetched.",
            )
        data = response.read(max_bytes + 1)
        content_type = _sample_url_response_header(response, "Content-Type")
    except urllib.error.HTTPError as exc:
        if 300 <= int(getattr(exc, "code", 0) or 0) < 400:
            raise HTTPException(
                status_code=400,
                detail="Sample URL redirects are not allowed.",
            ) from exc
        raise HTTPException(
            status_code=400,
            detail="Sample URL could not be fetched.",
        ) from exc
    except (OSError, urllib.error.URLError, http.client.HTTPException) as exc:
        raise HTTPException(
            status_code=400,
            detail="Sample URL could not be fetched.",
        ) from exc
    finally:
        if response is not None:
            _close_https_sample_url_response(response)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail={
                "reason": "brand_voice_sample_url_too_large",
                "max_bytes": max_bytes,
            },
        )
    return data, content_type, target.url


def _open_https_sample_url_request(
    target: _SampleUrlFetchTarget,
    *,
    timeout: int,
) -> Any:
    connection = _PINNED_HTTPS_CONNECTION_CLASS(
        target.host,
        port=target.port,
        connect_host=target.connect_host,
        timeout=timeout,
    )
    try:
        connection.request(
            "GET",
            target.path,
            headers={
                "Host": target.host_header,
                "User-Agent": _SAMPLE_URL_USER_AGENT,
                "Accept": "text/html,text/plain;q=0.9,*/*;q=0.5",
            },
        )
        response = connection.getresponse()
        return _PinnedSampleUrlResponse(response, connection)
    except Exception:
        connection.close()
        raise


def _close_https_sample_url_response(response: Any) -> None:
    close = getattr(response, "close", None)
    if callable(close):
        close()


def _sample_url_response_header(response: Any, name: str) -> str:
    getheader = getattr(response, "getheader", None)
    if callable(getheader):
        return str(getheader(name, "") or "")
    headers = getattr(response, "headers", None)
    if isinstance(headers, dict):
        return str(headers.get(name) or headers.get(name.lower()) or "")
    get = getattr(headers, "get", None)
    if callable(get):
        return str(get(name, "") or "")
    return ""


def _validate_https_sample_url_fetch_target(value: Any) -> _SampleUrlFetchTarget:
    url = _validate_https_sample_url(value)
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""
    port = parsed.port or 443
    try:
        connect_host = _validate_sample_url_host_resolution(host, port)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    path = urllib.parse.urlunparse((
        "",
        "",
        parsed.path or "/",
        parsed.params,
        parsed.query,
        "",
    ))
    return _SampleUrlFetchTarget(
        url=url,
        host=host,
        port=port,
        path=path,
        host_header=parsed.netloc,
        connect_host=connect_host,
    )


def _validate_https_sample_url(value: Any) -> str:
    text = str(value or "").strip()
    parsed = urllib.parse.urlparse(text)
    if parsed.scheme != "https" or not parsed.netloc:
        raise HTTPException(status_code=400, detail="Sample URL must be an https URL")
    try:
        parsed.port
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Sample URL must be an https URL",
        ) from exc
    host = parsed.hostname or ""
    if not host or _is_blocked_sample_url_host(host):
        raise HTTPException(status_code=400, detail="Sample URL host is not allowed")
    if parsed.username or parsed.password:
        raise HTTPException(
            status_code=400,
            detail="Sample URL must not include credentials",
        )
    return text


def _validate_sample_url_host_resolution(host: str, port: int) -> str:
    try:
        resolved = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ValueError("Sample URL host could not be resolved") from exc
    if not resolved:
        raise ValueError("Sample URL host could not be resolved")
    connect_host: str | None = None
    for item in resolved:
        sockaddr = item[4]
        if not sockaddr:
            raise ValueError("Sample URL host could not be resolved")
        address = str(sockaddr[0])
        if _is_blocked_sample_url_host(address):
            raise ValueError("Sample URL host is not allowed")
        if connect_host is None:
            connect_host = address
    if connect_host is None:
        raise ValueError("Sample URL host could not be resolved")
    return connect_host


def _is_blocked_sample_url_host(host: str) -> bool:
    lowered = host.strip().lower().rstrip(".")
    if lowered in {"localhost", "0.0.0.0"} or lowered.endswith(".local"):
        return True
    try:
        address = ipaddress.ip_address(lowered)
    except ValueError:
        return False
    return bool(
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
        or not address.is_global
    )


def _extract_readable_text(
    data: bytes,
    *,
    content_type: str,
) -> tuple[str | None, str]:
    text = data.decode(_sample_url_charset(content_type), errors="replace")
    if _looks_like_html(text, content_type):
        parser = _ReadableHtmlTextParser()
        parser.feed(text)
        parser.close()
        return parser.title, parser.text
    return None, _normalize_readable_text(text)


def _sample_url_charset(content_type: str) -> str:
    match = re.search(r"charset=([^;\s]+)", content_type, flags=re.IGNORECASE)
    if not match:
        return "utf-8"
    charset = match.group(1).strip('"').lower()
    try:
        "".encode(charset)
    except LookupError:
        return "utf-8"
    return charset


def _looks_like_html(text: str, content_type: str) -> bool:
    lowered_type = content_type.lower()
    if any(marker in lowered_type for marker in _HTML_CONTENT_TYPES):
        return True
    prefix = text.lstrip()[:200].lower()
    return prefix.startswith("<!doctype html") or prefix.startswith("<html")


def _normalize_readable_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


__all__ = [
    "BrandVoiceSampleUrlRequest",
    "BrandVoiceSampleUrlView",
    "BrandVoiceProfileView",
    "UpsertBrandVoiceProfileRequest",
    "create_content_ops_brand_voice_profiles_router",
]
