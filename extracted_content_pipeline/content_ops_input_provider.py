"""Input-provider contract for AI Content Ops request payloads.

The contract is deliberately small: ingestion layers may normalize their own
data, then hand Content Ops a package of request inputs without learning every
generator's service signature. The existing control surface remains the source
of truth for validating and planning generation.
"""

from __future__ import annotations

from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .campaign_ports import JsonDict, TenantScope


RequestPayload = Mapping[str, Any]

_REQUEST_KEYS = frozenset({
    "target_mode",
    "preset",
    "reasoning_preset",
    "outputs",
    "limit",
    "max_cost_usd",
    "account_usage_budget_usd",
    "account_usage_budget_days",
    "content_ops_cache_policy",
    "inputs",
    "ingestion_profile",
    "require_quality_gates",
    "allow_unimplemented_outputs",
})


@dataclass(frozen=True)
class ContentOpsInputPackage:
    """Normalized inputs produced by an upstream ingestion/provider layer.

    ``inputs`` is the only field that flows into ``ContentOpsRequest.inputs``.
    Request-level fields are defaults: explicit values from the caller's request
    payload win when ``merge_content_ops_input_package`` is used.
    """

    provider: str
    inputs: Mapping[str, Any]
    outputs: Sequence[str] = ()
    target_mode: str = "vendor_retention"
    ingestion_profile: str = "existing_evidence"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    warnings: Sequence[Mapping[str, Any]] = ()

    def as_dict(self) -> JsonDict:
        return {
            "provider": self.provider,
            "inputs": dict(self.inputs),
            "outputs": [str(output) for output in self.outputs],
            "target_mode": self.target_mode,
            "ingestion_profile": self.ingestion_profile,
            "metadata": dict(self.metadata),
            "warnings": [dict(warning) for warning in self.warnings],
        }


class ContentOpsInputProvider(Protocol):
    """Protocol for host-owned ingestion adapters that produce Content Ops inputs."""

    def build_content_ops_input_package(
        self,
        *,
        scope: TenantScope,
        request: RequestPayload | None = None,
    ) -> Awaitable[ContentOpsInputPackage] | ContentOpsInputPackage:
        """Return normalized inputs for one tenant-scoped Content Ops request."""


def content_ops_payload_from_input_package(
    package: ContentOpsInputPackage | Mapping[str, Any],
    *,
    request_payload: RequestPayload | None = None,
) -> JsonDict:
    """Return a plain payload accepted by ``request_from_mapping``.

    ``request_payload`` is optional caller/operator input. When present, its
    request-level fields and nested ``inputs`` values override provider defaults.
    """

    normalized_package = _normalize_package(package)
    base = _request_payload(request_payload or {})
    payload: JsonDict = {
        "target_mode": normalized_package.target_mode,
        "outputs": [str(output) for output in normalized_package.outputs],
        "inputs": dict(normalized_package.inputs),
        "ingestion_profile": normalized_package.ingestion_profile,
        "input_provider": _package_diagnostics(normalized_package),
    }
    for key, value in base.items():
        if value is None:
            continue
        if key == "inputs":
            payload["inputs"] = {
                **dict(payload["inputs"]),
                **_inputs_mapping(value),
            }
        elif key in _REQUEST_KEYS:
            payload[key] = value
    return _drop_empty_request_defaults(payload)


def merge_content_ops_input_package(
    request_payload: RequestPayload,
    package: ContentOpsInputPackage | Mapping[str, Any],
) -> JsonDict:
    """Merge provider inputs under an existing Content Ops request payload."""

    return content_ops_payload_from_input_package(
        package,
        request_payload=request_payload,
    )


def _normalize_package(
    package: ContentOpsInputPackage | Mapping[str, Any],
) -> ContentOpsInputPackage:
    if isinstance(package, ContentOpsInputPackage):
        return package
    if not isinstance(package, Mapping):
        raise TypeError("content ops input package must be a mapping or ContentOpsInputPackage")
    provider = _clean_text(package.get("provider")) or "unknown"
    return ContentOpsInputPackage(
        provider=provider,
        inputs=_inputs_mapping(package.get("inputs")),
        outputs=_string_tuple(package.get("outputs")),
        target_mode=_clean_text(package.get("target_mode")) or "vendor_retention",
        ingestion_profile=_clean_text(package.get("ingestion_profile"))
        or "existing_evidence",
        metadata=_metadata_mapping(package.get("metadata")),
        warnings=_warning_sequence(package.get("warnings")),
    )


def _request_payload(value: RequestPayload) -> JsonDict:
    if not isinstance(value, Mapping):
        raise TypeError("request_payload must be a mapping")
    return {
        str(key): item
        for key, item in value.items()
        if str(key) in _REQUEST_KEYS
    }


def _inputs_mapping(value: Any) -> JsonDict:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("content ops inputs must be a mapping")
    return {str(key): item for key, item in value.items()}


def _metadata_mapping(value: Any) -> JsonDict:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("content ops input package metadata must be a mapping")
    return {str(key): item for key, item in value.items()}


def _warning_sequence(value: Any) -> tuple[JsonDict, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError("content ops input package warnings must be a sequence")
    if not isinstance(value, Sequence):
        raise TypeError("content ops input package warnings must be a sequence")
    warnings: list[JsonDict] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError("content ops input package warnings must contain mappings")
        warnings.append({str(key): warning_value for key, warning_value in item.items()})
    return tuple(warnings)


def _package_diagnostics(package: ContentOpsInputPackage) -> JsonDict:
    return {
        "provider": package.provider,
        "metadata": dict(package.metadata),
        "warnings": [dict(warning) for warning in package.warnings],
    }


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    raise TypeError("content ops input package outputs must be a string or sequence")


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _drop_empty_request_defaults(payload: JsonDict) -> JsonDict:
    cleaned = dict(payload)
    if not cleaned.get("outputs"):
        cleaned.pop("outputs", None)
    if not cleaned.get("inputs"):
        cleaned.pop("inputs", None)
    return cleaned


__all__ = [
    "ContentOpsInputPackage",
    "ContentOpsInputProvider",
    "content_ops_payload_from_input_package",
    "merge_content_ops_input_package",
]
