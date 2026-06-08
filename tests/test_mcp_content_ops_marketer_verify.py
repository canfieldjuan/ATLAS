from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from unittest.mock import MagicMock

import pytest


class _MockToolManager:
    def __init__(self) -> None:
        self._tools = {}


class _MockFastMCP:
    def __init__(self, *args, **kwargs) -> None:
        self.settings = MagicMock()
        self._tool_manager = _MockToolManager()

    def tool(self, *args, **kwargs):
        def _register(fn):
            self._tool_manager._tools[fn.__name__] = fn
            return fn

        return _register

    def streamable_http_app(self):
        return object()

    def run(self, **kwargs) -> None:
        return None


sys.modules.setdefault("mcp", MagicMock())
sys.modules.setdefault("mcp.server", MagicMock())
_fastmcp_mod = MagicMock()
_fastmcp_mod.FastMCP = _MockFastMCP
sys.modules.setdefault("mcp.server.fastmcp", _fastmcp_mod)

from atlas_brain.config import settings
from atlas_brain.mcp import content_ops_marketer_verify_server as verify
from atlas_brain.mcp.auth import BearerAuthMiddleware
from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.claims_map import RegistryClaim
from extracted_content_pipeline.review_contract import RiskTier


VERIFY_TOOLS = {"verify_draft"}
DENIED_TOOLS = {
    "add_registry_claim",
    "approve",
    "fetch",
    "generate",
    "publish",
    "search",
    "unlock",
}


@dataclass
class _RegistryReader:
    scopes: list[TenantScope]

    async def list_registry_claims(self, *, scope: TenantScope):
        self.scopes.append(scope)
        return {
            "feature.sso": RegistryClaim(
                id="feature.sso",
                approved_wording="SSO is included on every plan",
                risk_tier=RiskTier.MEDIUM,
                expiration=date(2026, 12, 31),
            )
        }


@pytest.fixture(autouse=True)
def _reset_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(verify, "_registry_reader_override", None)
    monkeypatch.setattr(verify, "_account_resolver_override", None)


def _tool_names() -> set[str]:
    return set(verify.mcp._tool_manager._tools)


def _valid_payload() -> dict[str, object]:
    return {
        "asset_id": "asset-1",
        "rule_packet": {
            "brief": "brief-v1",
            "brand_voice": "voice-v1",
            "claim_registry": "claims-v1",
            "compliance": "compliance-v1",
            "channel_schema": "channel-v1",
        },
        "coverage": [
            {
                "rule_id": "VOICE-01",
                "requirement": "Rule must be evidenced",
                "status": "pass",
                "evidence": "quoted draft span",
            }
        ],
        "extracted_claims": [
            {
                "text": "SSO is included on every plan",
                "location": "hero",
                "registry_id": "feature.sso",
            }
        ],
        "as_of": "2026-06-08",
    }


def test_content_ops_marketer_verify_exposes_exact_tool_surface() -> None:
    assert _tool_names() == VERIFY_TOOLS
    assert _tool_names().isdisjoint(DENIED_TOOLS)


@pytest.mark.asyncio
async def test_verify_draft_fails_closed_without_account_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _RegistryReader(scopes=[])
    monkeypatch.setattr(verify, "_registry_reader_override", reader)
    monkeypatch.setattr(
        verify,
        "_account_resolver_override",
        verify.StaticContentOpsMarketerAccountResolver(" "),
    )

    payload = await verify.verify_draft(**_valid_payload())

    assert payload["ok"] is False
    assert payload["decision"] == "blocked"
    assert payload["reasons"] == ["tenant scope required"]
    assert reader.scopes == []


@pytest.mark.asyncio
async def test_verify_draft_delegates_to_bound_tenant_review_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _RegistryReader(scopes=[])
    monkeypatch.setattr(verify, "_registry_reader_override", reader)
    monkeypatch.setattr(
        verify,
        "_account_resolver_override",
        verify.StaticContentOpsMarketerAccountResolver(" acct-1 "),
    )

    payload = await verify.verify_draft(**_valid_payload())

    assert payload["ok"] is True
    assert payload["decision"] == "approved"
    assert payload["reasons"] == []
    assert payload["mapped_claims"][0]["status"] == "match"
    assert reader.scopes == [TenantScope(account_id="acct-1")]


@pytest.mark.asyncio
async def test_verify_draft_malformed_decoded_rows_block_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _RegistryReader(scopes=[])
    monkeypatch.setattr(verify, "_registry_reader_override", reader)
    monkeypatch.setattr(
        verify,
        "_account_resolver_override",
        verify.StaticContentOpsMarketerAccountResolver("acct-1"),
    )

    payload = await verify.verify_draft(
        asset_id=123,
        rule_packet={"brief": None},
        coverage=[{"rule_id": "VOICE-01", "status": "wat"}],
        extracted_claims=[{"text": None, "registry_id": 42}],
        comments=[{"category": "nit", "blocking": True}],
        as_of="not-a-date",
    )

    assert payload["ok"] is False
    assert payload["decision"] == "blocked"
    assert "rule packet not pinned" in payload["reasons"][0]
    assert "unresolved required coverage: VOICE-01" in payload["reasons"]
    assert "1 blocking comment(s)" in payload["reasons"]
    assert reader.scopes == [TenantScope(account_id="acct-1")]


@pytest.mark.asyncio
async def test_verify_draft_preserves_malformed_coverage_rows_as_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _RegistryReader(scopes=[])
    monkeypatch.setattr(verify, "_registry_reader_override", reader)
    monkeypatch.setattr(
        verify,
        "_account_resolver_override",
        verify.StaticContentOpsMarketerAccountResolver("acct-1"),
    )
    payload = _valid_payload()
    payload["coverage"] = [
        {
            "rule_id": "VOICE-01",
            "requirement": "Rule must be evidenced",
            "status": "pass",
            "evidence": "quoted draft span",
        },
        {"status": "pass", "evidence": "missing rule id should not disappear"},
        {"rule_id": 42, "status": "pass", "evidence": "non-string rule id"},
        "not-a-row",
    ]

    result = await verify.verify_draft(**payload)

    coverage = result["content_pr"]["coverage"]
    assert result["ok"] is False
    assert result["decision"] == "blocked"
    assert (
        "unresolved required coverage: "
        "MALFORMED-COVERAGE-2, MALFORMED-COVERAGE-3, MALFORMED-COVERAGE-4"
    ) in result["reasons"]
    assert [row["rule_id"] for row in coverage] == [
        "VOICE-01",
        "MALFORMED-COVERAGE-2",
        "MALFORMED-COVERAGE-3",
        "MALFORMED-COVERAGE-4",
    ]
    assert reader.scopes == [TenantScope(account_id="acct-1")]


def test_content_ops_marketer_verify_http_requires_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.mcp, "auth_token", "")

    with pytest.raises(RuntimeError, match="ATLAS_MCP_AUTH_TOKEN is required"):
        verify._streamable_http_app()


def test_content_ops_marketer_verify_http_wraps_with_bearer_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.mcp, "auth_token", "test-token-with-enough-entropy")

    assert isinstance(verify._streamable_http_app(), BearerAuthMiddleware)


@pytest.mark.parametrize("token", ["<token>", "token", "test-token"])
def test_content_ops_marketer_verify_http_rejects_bad_tokens(
    monkeypatch: pytest.MonkeyPatch,
    token: str,
) -> None:
    monkeypatch.setattr(settings.mcp, "auth_token", token)

    with pytest.raises(RuntimeError):
        verify._streamable_http_app()
