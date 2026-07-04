"""OpenAI-compatible LLM judge client for the fit pass (v2 S5, #1931).

A narrow, standalone client that POSTs one chat-completions request to an
OpenAI-compatible backend (OpenRouter or a local LM Studio server) and
turns the response into a FitDecision via the S2 parser. It deliberately
does NOT import atlas_brain, does NOT touch the B2B/global OpenRouter
keys, and uses stdlib urllib behind an injectable transport so tests fake
the HTTP boundary and CI never talks to a network.

Failure taxonomy (kept crisp for the S6 runner):
- config error (missing base_url/model, or missing key for a backend that
  needs one) -> RedditFitConfigError at build time (fail closed);
- transport / HTTP / non-JSON-envelope error -> FitClientError raised;
- the model replied but its content is not a valid FitDecision -> return
  (None, meta) with meta.parse_error set to a closed PARSE_ERROR_CODES
  value, so the runner drops it straight into a prediction envelope.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

from .config import FIT_BACKENDS, FIT_RESPONSE_FORMATS, RedditListeningSettings
from .fit import FIT_OUTPUT_JSON_SCHEMA, FitDecision, FitParseError, parse_fit_decision

# A transport is any callable that performs the POST and returns
# (status_code, response_text). The default uses stdlib urllib; tests pass
# a fake so no network is touched.
Transport = Callable[[str, dict, bytes, float], "tuple[int, str]"]

# Reasoning models (o1/o3/o4 families) reject temperature and use
# max_completion_tokens instead of max_tokens -- mirror the host
# OpenRouter adapter's dedicated branch.
_REASONING_MODEL_RE = re.compile(r"(?:^|/)o[1-9]")


class RedditFitConfigError(ValueError):
    """The fit backend is misconfigured (missing url/model/key). Fail
    closed at build time rather than at the first live call."""


class FitClientError(RuntimeError):
    """A judge call failed at the transport/HTTP/envelope level (network
    error, non-2xx, or a response that is not OpenAI-shaped). Distinct
    from a model that replied with malformed FitDecision content."""


@dataclass(frozen=True)
class FitCallMeta:
    """Provenance + usage for one judge call. ``parse_error`` is a closed
    PARSE_ERROR_CODES value when the model's content did not parse, else
    None."""

    model_id: str
    input_tokens: int
    output_tokens: int
    parse_error: str | None


def _default_transport(url: str, headers: dict, payload: bytes, timeout: float) -> tuple[int, str]:
    request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        return exc.code, body
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise FitClientError(f"fit judge transport error: {exc}") from exc


class OpenAICompatibleJudgeClient:
    """One backend behind the OpenAI /chat/completions contract. Reused
    for OpenRouter and local LM Studio -- only base_url/api_key/model and
    the structured-output strategy differ."""

    def __init__(
        self,
        *,
        backend: str,
        base_url: str,
        model: str,
        api_key: str,
        timeout_seconds: float,
        response_format: str = "json_schema",
        transport: Transport | None = None,
    ) -> None:
        self._backend = backend
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._response_format_mode = response_format
        self._transport = transport or _default_transport

    @property
    def model_id(self) -> str:
        return self._model

    def _response_format(self) -> dict | None:
        # Which structured-output mode a server accepts is a property of the
        # SERVER, not the backend name -- LM Studio needs json_schema (or
        # text) and rejects json_object; vLLM/OpenRouter take json_schema.
        # text returns None (no server-side constraint); the parser is the
        # authoritative gate in every mode.
        if self._response_format_mode == "json_schema":
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "fit_decision",
                    "strict": True,
                    "schema": FIT_OUTPUT_JSON_SCHEMA,
                },
            }
        if self._response_format_mode == "json_object":
            return {"type": "json_object"}
        return None

    def judge(self, messages: tuple[dict, ...]) -> tuple[FitDecision | None, FitCallMeta]:
        """Send one prompt, return the parsed decision (or None + a
        parse_error code when the model's content is malformed)."""
        body_obj = {
            "model": self._model,
            "messages": list(messages),
        }
        response_format = self._response_format()
        if response_format is not None:
            body_obj["response_format"] = response_format
        if _REASONING_MODEL_RE.search(self._model):
            body_obj["max_completion_tokens"] = 400
        else:
            body_obj["temperature"] = 0.0
            body_obj["max_tokens"] = 400
        payload = json.dumps(body_obj).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        status, body = self._transport(
            f"{self._base_url}/chat/completions", headers, payload, self._timeout
        )
        if not 200 <= status < 300:
            # Status only -- the provider body can echo the submitted prompt
            # (Reddit thread text) or diagnostics; keep it out of records.
            raise FitClientError(f"fit judge HTTP {status}")
        try:
            data = json.loads(body)
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage") or {}
            input_tokens = int(usage.get("prompt_tokens") or 0)
            output_tokens = int(usage.get("completion_tokens") or 0)
        except (
            json.JSONDecodeError, KeyError, IndexError, TypeError,
            ValueError, AttributeError,
        ) as exc:
            # Opaque: report only the failure kind, never provider content.
            raise FitClientError(
                f"fit judge response was not OpenAI-shaped ({type(exc).__name__})"
            ) from exc
        try:
            decision = parse_fit_decision(content)
        except FitParseError as exc:
            return None, FitCallMeta(
                model_id=self._model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                parse_error=exc.code,
            )
        return decision, FitCallMeta(
            model_id=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            parse_error=None,
        )


def build_judge_client(
    settings: RedditListeningSettings, *, transport: Transport | None = None
) -> OpenAICompatibleJudgeClient | None:
    """Build a judge client from settings, or return None when the backend
    is 'off'. Fails closed on a misconfigured backend so a live run cannot
    start half-configured."""
    backend = settings.fit_backend
    if backend not in FIT_BACKENDS:
        raise RedditFitConfigError(
            f"invalid fit_backend {backend!r}; allowed: {FIT_BACKENDS}"
        )
    if backend == "off":
        return None
    if not settings.fit_base_url:
        raise RedditFitConfigError(
            f"fit_backend={backend} requires ATLAS_REDDIT_FIT_BASE_URL"
        )
    if not settings.fit_model:
        raise RedditFitConfigError(
            f"fit_backend={backend} requires ATLAS_REDDIT_FIT_MODEL"
        )
    api_key = settings.fit_api_key.get_secret_value()
    if backend == "openrouter" and not api_key:
        # A local backend may run keyless; a hosted one must not be called
        # anonymously (and would leak nothing, but also would not work).
        raise RedditFitConfigError(
            "fit_backend=openrouter requires ATLAS_REDDIT_FIT_API_KEY"
        )
    if settings.fit_response_format not in FIT_RESPONSE_FORMATS:
        raise RedditFitConfigError(
            f"invalid fit_response_format {settings.fit_response_format!r}; "
            f"allowed: {FIT_RESPONSE_FORMATS}"
        )
    return OpenAICompatibleJudgeClient(
        backend=backend,
        base_url=settings.fit_base_url,
        model=settings.fit_model,
        api_key=api_key,
        timeout_seconds=settings.fit_timeout_seconds,
        response_format=settings.fit_response_format,
        transport=transport,
    )
