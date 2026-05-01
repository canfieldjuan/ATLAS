from __future__ import annotations

from datetime import datetime, timezone

import pytest

from extracted_content_pipeline.campaign_suppression import (
    CampaignSuppressionService,
    build_suppression_input,
    domain_from_email,
    normalize_domain,
    normalize_email,
)


class _SuppressionRepo:
    def __init__(self, suppressed: set[tuple[str | None, str | None]] | None = None):
        self.suppressed = suppressed or set()
        self.check_calls: list[dict] = []
        self.add_calls: list[dict] = []

    async def is_suppressed(self, *, email=None, domain=None):
        self.check_calls.append({"email": email, "domain": domain})
        return (email, domain) in self.suppressed

    async def add_suppression(self, **kwargs):
        self.add_calls.append(kwargs)


def test_normalize_email_trims_and_lowercases():
    assert normalize_email("  USER@Example.COM  ") == "user@example.com"
    assert normalize_email("   ") is None
    assert normalize_email(None) is None


def test_normalize_domain_trims_lowercases_and_accepts_at_prefix():
    assert normalize_domain("  @Example.COM.  ") == "example.com"
    assert normalize_domain(" . ") is None
    assert normalize_domain(None) is None


def test_domain_from_email_returns_none_for_invalid_email():
    assert domain_from_email("person@example.com") == "example.com"
    assert domain_from_email("not-an-email") is None
    assert domain_from_email("@example.com") is None
    assert domain_from_email("person@") is None


def test_build_suppression_input_normalizes_payload():
    expires_at = datetime(2026, 5, 1, tzinfo=timezone.utc)

    payload = build_suppression_input(
        email=" USER@Example.COM ",
        domain=" @Example.COM. ",
        reason=" unsubscribe ",
        source=" webhook ",
        campaign_id=" campaign-1 ",
        notes=" requested ",
        expires_at=expires_at,
        metadata={"provider": "resend"},
    )

    assert payload is not None
    assert payload.email == "user@example.com"
    assert payload.domain == "example.com"
    assert payload.reason == "unsubscribe"
    assert payload.source == "webhook"
    assert payload.campaign_id == "campaign-1"
    assert payload.notes == "requested"
    assert payload.expires_at == expires_at
    assert payload.metadata == {"provider": "resend"}


def test_build_suppression_input_returns_none_without_target():
    assert build_suppression_input(reason="unsubscribe") is None


def test_build_suppression_input_requires_reason():
    with pytest.raises(ValueError, match="reason is required"):
        build_suppression_input(email="person@example.com", reason=" ")


@pytest.mark.asyncio
async def test_service_checks_exact_email_before_domain_and_short_circuits():
    repo = _SuppressionRepo(suppressed={("person@example.com", None)})
    service = CampaignSuppressionService(repo)

    assert await service.is_suppressed(email=" PERSON@Example.COM ") is True

    assert repo.check_calls == [{"email": "person@example.com", "domain": None}]


@pytest.mark.asyncio
async def test_service_falls_back_to_email_domain_when_email_clear():
    repo = _SuppressionRepo(suppressed={(None, "example.com")})
    service = CampaignSuppressionService(repo)

    assert await service.is_suppressed(email="person@example.com") is True

    assert repo.check_calls == [
        {"email": "person@example.com", "domain": None},
        {"email": None, "domain": "example.com"},
    ]


@pytest.mark.asyncio
async def test_service_checks_explicit_domain_when_email_missing():
    repo = _SuppressionRepo(suppressed={(None, "example.com")})
    service = CampaignSuppressionService(repo)

    assert await service.is_suppressed(email=None, domain=" @Example.COM. ") is True

    assert repo.check_calls == [{"email": None, "domain": "example.com"}]


@pytest.mark.asyncio
async def test_service_returns_false_without_email_or_domain():
    repo = _SuppressionRepo()
    service = CampaignSuppressionService(repo)

    assert await service.is_suppressed(email=" ") is False
    assert repo.check_calls == []


@pytest.mark.asyncio
async def test_service_add_suppression_normalizes_and_calls_repository():
    repo = _SuppressionRepo()
    service = CampaignSuppressionService(repo)
    expires_at = datetime(2026, 5, 1, tzinfo=timezone.utc)

    added = await service.add_suppression(
        email=" Person@Example.COM ",
        reason=" complaint ",
        source=" webhook ",
        campaign_id=" campaign-1 ",
        notes=" user complained ",
        expires_at=expires_at,
        metadata={"provider": "resend"},
    )

    assert added is True
    assert repo.add_calls == [{
        "reason": "complaint",
        "email": "person@example.com",
        "domain": None,
        "source": "webhook",
        "campaign_id": "campaign-1",
        "notes": "user complained",
        "expires_at": expires_at,
        "metadata": {"provider": "resend"},
    }]


@pytest.mark.asyncio
async def test_service_add_suppression_skips_blank_target():
    repo = _SuppressionRepo()
    service = CampaignSuppressionService(repo)

    assert await service.add_suppression(email=" ", domain=None, reason="unsubscribe") is False
    assert repo.add_calls == []
