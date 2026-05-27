"""Content Ops LLM cache policy decisions.

This module decides whether a Content Ops LLM call is eligible for a future
exact-cache lookup/store. It does not perform cache I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal


ContentOpsCacheMode = Literal["exact", "no_store"]

DEFAULT_CACHEABLE_ASSET_TYPES = (
    "blog_post",
    "landing_page",
    "sales_brief",
    "email_campaign",
    "report",
)
CUSTOMER_DATA_SOURCE_MARKERS = (
    "support_ticket",
    "support_tickets",
    "customer_data",
    "customer_upload",
)
NO_STORE_POLICY_VALUES = {"off", "false", "no", "none", "no_store", "no-store"}
EXACT_POLICY_VALUES = {"exact", "exact_cache", "exact-cache"}


@dataclass(frozen=True)
class ContentOpsCacheDecision:
    """Decision metadata for one Content Ops LLM call."""

    mode: ContentOpsCacheMode
    reason: str
    namespace: str | None = None
    account_id: str | None = None

    @property
    def cacheable(self) -> bool:
        return self.mode == "exact"

    def trace_metadata(self) -> dict[str, str]:
        metadata = {
            "cache_mode": self.mode,
            "cache_reason": self.reason,
        }
        if self.namespace:
            metadata["cache_namespace"] = self.namespace
        if self.account_id:
            metadata["cache_account_id"] = self.account_id
        return metadata


@dataclass(frozen=True)
class ContentOpsExactCachePolicy:
    """Content Ops exact-cache eligibility policy.

    The policy is conservative by design. Exact cache starts disabled, requires
    tenant account scope, and refuses customer-upload/support-ticket markers.
    """

    exact_cache_enabled: bool = False
    customer_data_exact_cache_enabled: bool = False
    namespace_prefix: str = "content_ops"
    cacheable_asset_types: tuple[str, ...] = DEFAULT_CACHEABLE_ASSET_TYPES

    @classmethod
    def from_settings(cls, settings_obj: Any) -> "ContentOpsExactCachePolicy":
        return cls(
            exact_cache_enabled=bool(
                getattr(settings_obj, "exact_cache_enabled", False),
            ),
            customer_data_exact_cache_enabled=bool(
                getattr(settings_obj, "customer_data_exact_cache_enabled", False),
            ),
            namespace_prefix=_clean_text(
                getattr(settings_obj, "exact_cache_namespace_prefix", None),
            )
            or "content_ops",
        )

    def decide(
        self,
        metadata: Mapping[str, Any] | None,
        *,
        messages: Sequence[Any] | None = None,
    ) -> ContentOpsCacheDecision:
        del messages  # Future adapter may use a redacted/digest-only envelope.
        values = dict(metadata or {})
        requested_policy = _policy_value(
            values.get("content_ops_cache_policy") or values.get("cache_policy")
        )
        asset_type = _clean_text(values.get("asset_type"))
        account_id = _clean_text(values.get("account_id"))

        if requested_policy in NO_STORE_POLICY_VALUES:
            return ContentOpsCacheDecision("no_store", "policy_no_store")
        if not self.exact_cache_enabled:
            return ContentOpsCacheDecision("no_store", "exact_cache_disabled")
        if not requested_policy:
            return ContentOpsCacheDecision("no_store", "policy_no_store")
        if requested_policy and requested_policy not in EXACT_POLICY_VALUES:
            return ContentOpsCacheDecision("no_store", "unsupported_cache_policy")
        if not account_id:
            return ContentOpsCacheDecision("no_store", "missing_account_scope")
        if not asset_type or asset_type not in self.cacheable_asset_types:
            return ContentOpsCacheDecision("no_store", "unsupported_asset_type")
        if (
            not self.customer_data_exact_cache_enabled
            and _has_customer_data_marker(values)
        ):
            return ContentOpsCacheDecision("no_store", "customer_data_no_store")

        namespace = f"{self.namespace_prefix}.{asset_type}"
        return ContentOpsCacheDecision(
            "exact",
            "eligible",
            namespace=namespace,
            account_id=account_id,
        )


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _policy_value(value: Any) -> str:
    return _clean_text(value).lower().replace(" ", "_")


def _has_customer_data_marker(metadata: Mapping[str, Any]) -> bool:
    candidates = (
        metadata.get("source_type"),
        metadata.get("source_types"),
        metadata.get("source_material_type"),
        metadata.get("source_material_types"),
        metadata.get("input_provider"),
        metadata.get("provider"),
    )
    markers = {_clean_text(marker).lower() for marker in CUSTOMER_DATA_SOURCE_MARKERS}
    return any(_value_contains_marker(value, markers) for value in candidates)


def _value_contains_marker(value: Any, markers: set[str]) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in markers or any(
            token.strip() in markers for token in normalized.split(",")
        ) or any(marker in normalized for marker in markers)
    if isinstance(value, Mapping):
        return any(_value_contains_marker(item, markers) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return any(_value_contains_marker(item, markers) for item in value)
    return False
