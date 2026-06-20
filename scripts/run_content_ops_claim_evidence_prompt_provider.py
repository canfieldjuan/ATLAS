#!/usr/bin/env python3
"""Run claim/evidence prompt packets through an OpenAI-compatible provider."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urlparse

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.claim_evidence_benchmark import (
    ClaimEvidencePromptPacket,
    run_claim_evidence_prompt_packets,
)


FORMAT_AUTO = "auto"
ROW_FORMAT_JSON = "json"
ROW_FORMAT_JSONL = "jsonl"
VALID_ROW_FORMATS = (FORMAT_AUTO, ROW_FORMAT_JSON, ROW_FORMAT_JSONL)
DEFAULT_API_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_TOKENS = 320
TOKEN_LIMIT_AUTO = "auto"
TOKEN_LIMIT_MAX_TOKENS = "max_tokens"
TOKEN_LIMIT_MAX_COMPLETION_TOKENS = "max_completion_tokens"
VALID_TOKEN_LIMIT_FIELDS = (
    TOKEN_LIMIT_AUTO,
    TOKEN_LIMIT_MAX_TOKENS,
    TOKEN_LIMIT_MAX_COMPLETION_TOKENS,
)
SCHEMA_MODE_AUTO = "auto"
SCHEMA_MODE_CANONICAL = "canonical"
SCHEMA_MODE_COMPATIBLE = "compatible"
VALID_SCHEMA_MODES = (SCHEMA_MODE_AUTO, SCHEMA_MODE_CANONICAL, SCHEMA_MODE_COMPATIBLE)
ACCEPTABLE_FINISH_REASONS = frozenset({"stop"})
UNSUPPORTED_STRICT_SCHEMA_KEYS = frozenset(
    {
        "format",
        "maxItems",
        "maxLength",
        "maximum",
        "minItems",
        "minLength",
        "minimum",
        "multipleOf",
        "pattern",
        "patternProperties",
    }
)


class ProviderConfigurationError(RuntimeError):
    """Provider configuration is missing or malformed."""


class ProviderHTTPStatusError(RuntimeError):
    """Provider returned a non-success HTTP status."""


class ProviderResponseDecodeError(RuntimeError):
    """Provider returned malformed JSON."""


class ProviderResponseShapeError(RuntimeError):
    """Provider returned an unexpected completion envelope."""


class ProviderResponseJSONError(RuntimeError):
    """Provider message content was not valid JSON."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Content Ops claim/evidence prompt packets against an OpenAI-compatible chat-completions provider."
    )
    parser.add_argument("packets_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument(
        "--packets-format",
        choices=VALID_ROW_FORMATS,
        default=FORMAT_AUTO,
        help="Prompt packet format. Auto accepts only .json and .jsonl suffixes.",
    )
    parser.add_argument(
        "--output-format",
        choices=VALID_ROW_FORMATS,
        default=FORMAT_AUTO,
        help="Response-row format. Auto accepts only .json and .jsonl suffixes.",
    )
    parser.add_argument(
        "--api-base-url",
        default=os.environ.get("ATLAS_CLAIM_EVIDENCE_OPENAI_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or DEFAULT_API_BASE_URL,
        help="OpenAI-compatible API base URL, without /chat/completions.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the provider API key.",
    )
    parser.add_argument(
        "--stability-run-count",
        type=int,
        default=0,
        help="Additional stability reruns per packet.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--token-limit-field",
        choices=VALID_TOKEN_LIMIT_FIELDS,
        default=TOKEN_LIMIT_AUTO,
        help="Token limit field to send. Auto uses max_completion_tokens for OpenAI o-series model ids.",
    )
    parser.add_argument(
        "--schema-mode",
        choices=VALID_SCHEMA_MODES,
        default=SCHEMA_MODE_AUTO,
        help="Structured-output schema mode. Auto strips provider-unsupported constraints for Azure or fine-tuned model ids.",
    )
    parser.add_argument(
        "--store-completions",
        action="store_true",
        help="Allow providers that support storage to retain completions and attach run metadata.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser


def _infer_row_format(
    path: Path,
    requested_format: str,
    label: str,
) -> tuple[str | None, str | None]:
    if requested_format not in VALID_ROW_FORMATS:
        return None, f"{label} format must be auto, json, or jsonl"
    if requested_format != FORMAT_AUTO:
        return requested_format, None
    suffix = path.suffix.lower()
    if suffix == ".json":
        return ROW_FORMAT_JSON, None
    if suffix == ".jsonl":
        return ROW_FORMAT_JSONL, None
    return None, f"{label} format auto-detection requires .json or .jsonl suffix"


def _read_text(path: Path, label: str) -> tuple[str | None, str | None]:
    try:
        return path.read_text(encoding="utf-8"), None
    except FileNotFoundError:
        return None, f"{label} file not found: {path}"
    except IsADirectoryError:
        return None, f"{label} path is a directory: {path}"
    except UnicodeDecodeError as error:
        return None, f"{label} file could not be read: {path}: {error}"
    except OSError as error:
        return None, f"{label} file could not be read: {path}: {error.strerror or error}"


def _decode_row_text(
    text: object,
    row_format: str,
    label: str,
) -> tuple[tuple[Mapping[str, object], ...], tuple[str, ...]]:
    if not isinstance(text, str):
        return (), (f"{label} text must be a string",)
    if row_format == ROW_FORMAT_JSON:
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as error:
            return (), (f"{label} json is malformed: {error.msg}",)
        if not isinstance(decoded, list):
            return (), (f"{label} json must decode to an array",)
        return _typed_rows(decoded, label)
    if row_format == ROW_FORMAT_JSONL:
        rows: list[object] = []
        errors: list[str] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                errors.append(f"{label} jsonl line {line_number} is malformed: {error.msg}")
        if errors:
            return (), tuple(errors)
        return _typed_rows(rows, label)
    return (), (f"{label} format must be json or jsonl",)


def _typed_rows(
    rows: Sequence[object],
    label: str,
) -> tuple[tuple[Mapping[str, object], ...], tuple[str, ...]]:
    typed_rows: list[Mapping[str, object]] = []
    errors: list[str] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            errors.append(f"{label} row {index} must be an object")
            continue
        typed_rows.append(row)
    return tuple(typed_rows), tuple(errors)


def _chat_completions_url(api_base_url: str) -> str:
    if not isinstance(api_base_url, str) or not api_base_url.strip():
        raise ProviderConfigurationError()
    base = api_base_url.rstrip("/")
    parsed = urlparse(base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ProviderConfigurationError()
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _strip_unsupported_schema_constraints(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_unsupported_schema_constraints(item)
            for key, item in value.items()
            if key not in UNSUPPORTED_STRICT_SCHEMA_KEYS
        }
    if isinstance(value, list):
        return [_strip_unsupported_schema_constraints(item) for item in value]
    return value


def _is_fine_tuned_model(model_id: str) -> bool:
    normalized = model_id.strip().lower()
    return normalized.startswith("ft:") or ":ft:" in normalized


def _requires_compatible_schema(api_base_url: str, model_id: str) -> bool:
    return "azure" in api_base_url.lower() or _is_fine_tuned_model(model_id)


def _schema_for_provider(
    packet: ClaimEvidencePromptPacket,
    *,
    api_base_url: str,
    schema_mode: str,
) -> Mapping[str, object]:
    if schema_mode == SCHEMA_MODE_COMPATIBLE or (
        schema_mode == SCHEMA_MODE_AUTO
        and _requires_compatible_schema(api_base_url, packet.model_id)
    ):
        sanitized = _strip_unsupported_schema_constraints(packet.response_schema)
        return sanitized if isinstance(sanitized, Mapping) else packet.response_schema
    return packet.response_schema


def _response_format(
    packet: ClaimEvidencePromptPacket,
    *,
    api_base_url: str,
    schema_mode: str,
) -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "claim_evidence_response",
            "strict": True,
            "schema": _schema_for_provider(
                packet,
                api_base_url=api_base_url,
                schema_mode=schema_mode,
            ),
        },
    }


def _extract_content_json(data: object) -> Mapping[str, object]:
    if not isinstance(data, Mapping):
        raise ProviderResponseShapeError()
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderResponseShapeError()
    first = choices[0]
    if not isinstance(first, Mapping):
        raise ProviderResponseShapeError()
    if first.get("finish_reason") not in ACCEPTABLE_FINISH_REASONS:
        raise ProviderResponseShapeError()
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise ProviderResponseShapeError()
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ProviderResponseShapeError()
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError as error:
        raise ProviderResponseJSONError() from error
    if not isinstance(decoded, Mapping):
        raise ProviderResponseJSONError()
    return decoded


def _uses_max_completion_tokens(model_id: str) -> bool:
    normalized = model_id.strip().lower()
    if normalized.startswith(("o1", "o3", "o4")):
        return True
    return any(marker in normalized for marker in ("/o1", "/o3", "/o4"))


def _token_limit_field(model_id: str, requested_field: str) -> str:
    if requested_field == TOKEN_LIMIT_AUTO:
        if _uses_max_completion_tokens(model_id):
            return TOKEN_LIMIT_MAX_COMPLETION_TOKENS
        return TOKEN_LIMIT_MAX_TOKENS
    return requested_field


def _build_openai_compatible_provider(
    *,
    api_base_url: str,
    api_key: str,
    timeout_seconds: float,
    max_tokens: int,
    token_limit_field: str,
    schema_mode: str,
    store_completions: bool,
    temperature: float,
    client: httpx.Client | None = None,
):
    if not isinstance(api_key, str) or not api_key.strip():
        raise ProviderConfigurationError()
    if timeout_seconds <= 0:
        raise ProviderConfigurationError()
    if max_tokens <= 0:
        raise ProviderConfigurationError()
    if token_limit_field not in VALID_TOKEN_LIMIT_FIELDS:
        raise ProviderConfigurationError()
    if schema_mode not in VALID_SCHEMA_MODES:
        raise ProviderConfigurationError()

    own_client = client is None
    active_client = client or httpx.Client(timeout=timeout_seconds)
    endpoint = _chat_completions_url(api_base_url)

    def provider(
        packet: ClaimEvidencePromptPacket,
        run_type: str,
        run_id: str,
    ) -> Mapping[str, object]:
        metadata = {"run_type": run_type}
        if run_id:
            metadata["run_id"] = run_id
        limit_field = _token_limit_field(packet.model_id, token_limit_field)
        payload: dict[str, object] = {
            "model": packet.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict claim/evidence witness. Return only "
                        "JSON matching the supplied schema."
                    ),
                },
                {"role": "user", "content": packet.prompt},
            ],
            "response_format": _response_format(
                packet,
                api_base_url=api_base_url,
                schema_mode=schema_mode,
            ),
            limit_field: max_tokens,
            "temperature": temperature,
            "store": bool(store_completions),
        }
        if store_completions:
            payload["metadata"] = metadata
        try:
            response = active_client.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        except httpx.HTTPError as error:
            raise ProviderHTTPStatusError() from error
        if response.status_code < 200 or response.status_code >= 300:
            raise ProviderHTTPStatusError()
        try:
            return _extract_content_json(response.json())
        except json.JSONDecodeError as error:
            raise ProviderResponseDecodeError() from error

    provider.close = active_client.close if own_client else lambda: None  # type: ignore[attr-defined]
    return provider


def _render_rows(rows: Sequence[Mapping[str, object]], output_format: str) -> str:
    if output_format == ROW_FORMAT_JSONL:
        return "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    return json.dumps(list(rows), indent=2, sort_keys=True) + "\n"


def _write_rows(path: Path, content: str) -> str | None:
    if path.is_symlink():
        return f"output path is a symlink: {path}"
    if path.exists() and path.is_dir():
        return f"output path is a directory: {path}"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as error:
        return f"output file could not be written: {path}: {error.strerror or error}"
    return None


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve(strict=False) == right.resolve(strict=False)
    except OSError:
        return left.absolute() == right.absolute()


def _error_payload(
    errors: Sequence[str],
    *,
    output_path: Path | None = None,
    output_format: str | None = None,
    response_count: int = 0,
) -> dict[str, Any]:
    return {
        "ok": False,
        "output_path": str(output_path) if output_path is not None else "",
        "output_format": output_format or "",
        "response_count": response_count,
        "errors": list(errors),
    }


def run_prompt_packets_with_openai_compatible_provider(
    packets_path: Path,
    output_path: Path,
    *,
    packets_format: str = FORMAT_AUTO,
    output_format: str = FORMAT_AUTO,
    api_base_url: str = DEFAULT_API_BASE_URL,
    api_key_env: str = "OPENAI_API_KEY",
    stability_run_count: int = 0,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    token_limit_field: str = TOKEN_LIMIT_AUTO,
    schema_mode: str = SCHEMA_MODE_AUTO,
    store_completions: bool = False,
    temperature: float = 0.0,
    client: httpx.Client | None = None,
) -> tuple[int, dict[str, Any]]:
    packet_format, packet_format_error = _infer_row_format(
        packets_path,
        packets_format,
        "packets",
    )
    if packet_format_error:
        return 2, _error_payload((packet_format_error,), output_path=output_path)
    response_format, response_format_error = _infer_row_format(
        output_path,
        output_format,
        "output",
    )
    if response_format_error:
        return 2, _error_payload((response_format_error,), output_path=output_path)
    if _same_path(output_path, packets_path):
        return 2, _error_payload(
            ("output path must differ from packets path",),
            output_path=output_path,
            output_format=response_format,
        )

    packets_text, read_error = _read_text(packets_path, "packets")
    if read_error:
        return 2, _error_payload(
            (read_error,),
            output_path=output_path,
            output_format=response_format,
        )
    packet_rows, decode_errors = _decode_row_text(
        packets_text,
        packet_format or "",
        "packets",
    )
    if decode_errors:
        return 1, _error_payload(
            decode_errors,
            output_path=output_path,
            output_format=response_format,
        )

    api_key = os.environ.get(api_key_env, "")
    try:
        provider = _build_openai_compatible_provider(
            api_base_url=api_base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
            token_limit_field=token_limit_field,
            schema_mode=schema_mode,
            store_completions=store_completions,
            temperature=temperature,
            client=client,
        )
    except ProviderConfigurationError:
        return 2, _error_payload(
            (f"{api_key_env} missing or provider options invalid",),
            output_path=output_path,
            output_format=response_format,
        )

    try:
        run = run_claim_evidence_prompt_packets(
            packet_rows,
            provider,
            stability_run_count=stability_run_count,
        )
    finally:
        provider.close()  # type: ignore[attr-defined]
    if not run.ok:
        return 1, _error_payload(
            run.errors,
            output_path=output_path,
            output_format=response_format,
        )

    write_error = _write_rows(output_path, _render_rows(run.rows, response_format))
    if write_error:
        return 2, _error_payload(
            (write_error,),
            output_path=output_path,
            output_format=response_format,
            response_count=len(run.rows),
        )

    return (
        0,
        {
            "ok": True,
            "output_path": str(output_path),
            "output_format": response_format,
            "response_count": len(run.rows),
            "errors": [],
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    exit_code, payload = run_prompt_packets_with_openai_compatible_provider(
        args.packets_path,
        args.output_path,
        packets_format=args.packets_format,
        output_format=args.output_format,
        api_base_url=args.api_base_url,
        api_key_env=args.api_key_env,
        stability_run_count=args.stability_run_count,
        timeout_seconds=args.timeout_seconds,
        max_tokens=args.max_tokens,
        token_limit_field=args.token_limit_field,
        schema_mode=args.schema_mode,
        store_completions=args.store_completions,
        temperature=args.temperature,
    )
    print(json.dumps(payload, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
