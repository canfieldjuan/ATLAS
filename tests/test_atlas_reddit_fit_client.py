"""OpenAI-compatible judge client tests (v2 S5, #1931).

The HTTP/model boundary is the one true external surface, faked here via
an injectable transport -- CI never touches a network. Everything else is
real: real settings, real parser, real config validation.
"""

from __future__ import annotations

import json

import pytest

from atlas_reddit.config import RedditListeningSettings
from atlas_reddit.fit import FitDecision
from atlas_reddit.fit_client import (
    FitClientError,
    OpenAICompatibleJudgeClient,
    RedditFitConfigError,
    build_judge_client,
)

_MESSAGES = (
    {"role": "system", "content": "judge fit"},
    {"role": "user", "content": "Candidate thread: ..."},
)


class FakeTransport:
    """Records the last request and returns a canned (status, body)."""

    def __init__(self, status: int = 200, body: str | None = None) -> None:
        self.status = status
        self.body = body if body is not None else _ok_body()
        self.calls: list[dict] = []

    def __call__(self, url: str, headers: dict, payload: bytes, timeout: float):
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "payload": json.loads(payload.decode("utf-8")),
                "timeout": timeout,
            }
        )
        return self.status, self.body


def _ok_body(**prediction_overrides) -> str:
    prediction = {
        "verdict": "yes",
        "reason": "Repeat questions despite docs.",
        "angle": "Ask what the ticket history shows.",
        "risk_flags": [],
    }
    prediction.update(prediction_overrides)
    return json.dumps(
        {
            "choices": [{"message": {"content": json.dumps(prediction)}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 40},
        }
    )


def _settings(**overrides) -> RedditListeningSettings:
    base = dict(
        fit_backend="openrouter",
        fit_base_url="https://openrouter.ai/api/v1",
        fit_model="anthropic/claude-3.5-sonnet",
        fit_api_key="sk-fit-xxx",
        fit_timeout_seconds=15.0,
    )
    base.update(overrides)
    return RedditListeningSettings(_env_file=None, **base)


# -- build_judge_client: fail closed ----------------------------------------


def test_off_backend_builds_no_client() -> None:
    assert build_judge_client(_settings(fit_backend="off")) is None


def test_invalid_backend_fails_closed() -> None:
    # bypass pydantic-free construction path: settings hold any string
    s = _settings(fit_backend="off")
    object.__setattr__(s, "fit_backend", "sideways")
    with pytest.raises(RedditFitConfigError, match="invalid fit_backend"):
        build_judge_client(s)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"fit_base_url": ""}, "FIT_BASE_URL"),
        ({"fit_model": ""}, "FIT_MODEL"),
        ({"fit_api_key": ""}, "FIT_API_KEY"),
    ],
)
def test_openrouter_requires_url_model_key(overrides: dict, match: str) -> None:
    with pytest.raises(RedditFitConfigError, match=match):
        build_judge_client(_settings(**overrides))


def test_local_backend_may_run_keyless() -> None:
    client = build_judge_client(
        _settings(
            fit_backend="local",
            fit_base_url="http://127.0.0.1:1234/v1",
            fit_model="local-model",
            fit_api_key="",
        )
    )
    assert client is not None


# -- judge(): happy path + usage --------------------------------------------


def test_judge_returns_decision_and_usage() -> None:
    transport = FakeTransport()
    client = build_judge_client(_settings(), transport=transport)
    decision, meta = client.judge(_MESSAGES)
    assert isinstance(decision, FitDecision)
    assert decision.verdict == "yes"
    assert (meta.input_tokens, meta.output_tokens) == (120, 40)
    assert meta.parse_error is None
    assert meta.model_id == "anthropic/claude-3.5-sonnet"


def test_request_carries_no_reddit_credentials() -> None:
    """The fit request must never contain Reddit creds, and the bearer key
    is the FIT key -- not the B2B/global OpenRouter key."""
    transport = FakeTransport()
    settings = _settings(
        client_id="reddit-client-id",
        client_secret="reddit-secret",
        refresh_token="reddit-refresh",
        username="reddit-user",
    )
    client = build_judge_client(settings, transport=transport)
    client.judge(_MESSAGES)
    call = transport.calls[0]
    blob = json.dumps(call["payload"]) + json.dumps(call["headers"])
    for secret in ("reddit-client-id", "reddit-secret", "reddit-refresh", "reddit-user"):
        assert secret not in blob
    assert call["headers"]["Authorization"] == "Bearer sk-fit-xxx"
    assert call["timeout"] == 15.0
    assert call["url"] == "https://openrouter.ai/api/v1/chat/completions"


def test_default_response_format_is_json_schema_for_both_backends() -> None:
    """json_schema is the default for local AND openrouter -- LM Studio,
    vLLM and OpenRouter all accept it; the old local=json_object default
    was rejected by LM Studio (400)."""
    or_transport = FakeTransport()
    build_judge_client(_settings(), transport=or_transport).judge(_MESSAGES)
    assert or_transport.calls[0]["payload"]["response_format"]["type"] == "json_schema"

    local_transport = FakeTransport()
    build_judge_client(
        _settings(
            fit_backend="local",
            fit_base_url="http://127.0.0.1:1234/v1",
            fit_model="m",
            fit_api_key="",
        ),
        transport=local_transport,
    ).judge(_MESSAGES)
    assert local_transport.calls[0]["payload"]["response_format"]["type"] == "json_schema"
    # keyless local: no Authorization header
    assert "Authorization" not in local_transport.calls[0]["headers"]


def test_response_format_json_object_mode() -> None:
    t = FakeTransport()
    build_judge_client(
        _settings(fit_response_format="json_object"), transport=t
    ).judge(_MESSAGES)
    assert t.calls[0]["payload"]["response_format"] == {"type": "json_object"}


def test_response_format_text_mode_omits_the_key() -> None:
    """text mode sends no server-side constraint -- for servers that support
    neither json_schema nor json_object; the parser still gates."""
    t = FakeTransport()
    build_judge_client(
        _settings(fit_response_format="text"), transport=t
    ).judge(_MESSAGES)
    assert "response_format" not in t.calls[0]["payload"]


def test_invalid_response_format_fails_closed() -> None:
    with pytest.raises(RedditFitConfigError, match="fit_response_format"):
        build_judge_client(_settings(fit_response_format="yaml"))


# -- judge(): failure taxonomy ----------------------------------------------


def test_malformed_model_content_is_parse_error_not_raise() -> None:
    body = json.dumps(
        {
            "choices": [{"message": {"content": "the thread looks fit to me"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
    )
    client = build_judge_client(_settings(), transport=FakeTransport(body=body))
    decision, meta = client.judge(_MESSAGES)
    assert decision is None
    assert meta.parse_error == "model_output_invalid_json"
    assert (meta.input_tokens, meta.output_tokens) == (10, 5)


def test_schema_violation_content_is_parse_error() -> None:
    body = _ok_body(verdict="definitely")  # not in the enum
    client = build_judge_client(_settings(), transport=FakeTransport(body=body))
    decision, meta = client.judge(_MESSAGES)
    assert decision is None
    assert meta.parse_error == "model_output_schema_mismatch"


def test_http_non_2xx_raises_client_error_without_echoing_body() -> None:
    # the provider body can echo the submitted Reddit prompt or diagnostics;
    # the error must carry status only, never that content.
    secret_body = "error: your prompt was 'jane.doe@example.com asked about docs'"
    client = build_judge_client(
        _settings(), transport=FakeTransport(status=429, body=secret_body)
    )
    with pytest.raises(FitClientError, match="HTTP 429") as excinfo:
        client.judge(_MESSAGES)
    assert "jane.doe@example.com" not in str(excinfo.value)
    assert "prompt" not in str(excinfo.value)


@pytest.mark.parametrize(
    "usage",
    [
        "not-a-dict",
        {"prompt_tokens": "lots", "completion_tokens": 5},
        {"prompt_tokens": [1, 2], "completion_tokens": 5},
    ],
)
def test_malformed_usage_is_client_error_not_traceback(usage) -> None:
    """Valid content but a malformed usage object is a non-OpenAI-shaped
    envelope -> clean FitClientError, never a raw AttributeError/ValueError
    reaching the S6 runner. The message is opaque (no provider content)."""
    body = json.dumps(
        {
            "choices": [{"message": {"content": json.dumps(
                {"verdict": "yes", "reason": "r", "angle": "a", "risk_flags": []}
            )}}],
            "usage": usage,
        }
    )
    client = build_judge_client(_settings(), transport=FakeTransport(body=body))
    with pytest.raises(FitClientError, match="not OpenAI-shaped") as excinfo:
        client.judge(_MESSAGES)
    assert "lots" not in str(excinfo.value)  # opaque: no provider value echoed


def test_reasoning_model_omits_temperature_uses_completion_tokens() -> None:
    """o1/o3/o4 models reject temperature and use max_completion_tokens;
    fit_model is an unconstrained operator setting, so the client adapts."""
    transport = FakeTransport()
    build_judge_client(
        _settings(fit_model="openai/o1-preview"), transport=transport
    ).judge(_MESSAGES)
    payload = transport.calls[0]["payload"]
    assert payload["max_completion_tokens"] == 400
    assert "temperature" not in payload
    assert "max_tokens" not in payload

    # a normal model keeps the standard params
    normal = FakeTransport()
    build_judge_client(_settings(), transport=normal).judge(_MESSAGES)
    assert normal.calls[0]["payload"]["temperature"] == 0.0
    assert normal.calls[0]["payload"]["max_tokens"] == 400
    assert "max_completion_tokens" not in normal.calls[0]["payload"]


def test_non_openai_shaped_response_raises_client_error() -> None:
    client = build_judge_client(
        _settings(), transport=FakeTransport(body='{"unexpected": true}')
    )
    with pytest.raises(FitClientError, match="not OpenAI-shaped"):
        client.judge(_MESSAGES)


def test_transport_error_surfaces_as_client_error() -> None:
    def boom(url, headers, payload, timeout):
        raise FitClientError("fit judge transport error: connection refused")

    client = OpenAICompatibleJudgeClient(
        backend="local",
        base_url="http://127.0.0.1:1234/v1",
        model="m",
        api_key="",
        timeout_seconds=5.0,
        transport=boom,
    )
    with pytest.raises(FitClientError, match="transport error"):
        client.judge(_MESSAGES)


# -- purity -----------------------------------------------------------------


def test_client_has_no_atlas_brain_or_network_imports() -> None:
    from pathlib import Path

    import ast

    source = (
        Path(__file__).parent.parent / "atlas_reddit" / "fit_client.py"
    ).read_text(encoding="utf-8")
    absolute: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            absolute.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            absolute.add(node.module.split(".")[0])
    # only stdlib absolute imports; intra-package relative imports (level>0)
    # are the standalone package itself
    assert absolute <= {"__future__", "json", "re", "urllib", "dataclasses", "typing"}, absolute
    assert "atlas_brain" not in absolute
    assert "requests" not in absolute and "httpx" not in absolute
