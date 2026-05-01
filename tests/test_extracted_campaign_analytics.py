from __future__ import annotations

import pytest

from extracted_content_pipeline.campaign_analytics import CampaignAnalyticsRefreshService


class _CampaignRepo:
    def __init__(self, *, error: Exception | None = None):
        self.error = error
        self.refresh_calls = 0

    async def refresh_analytics(self):
        self.refresh_calls += 1
        if self.error:
            raise self.error

    async def save_drafts(self, drafts, *, scope):  # pragma: no cover - protocol filler
        raise AssertionError("not used")

    async def list_due_sends(self, *, limit, now):  # pragma: no cover - protocol filler
        raise AssertionError("not used")

    async def mark_sent(self, *, campaign_id, result, sent_at):  # pragma: no cover
        raise AssertionError("not used")

    async def mark_cancelled(self, *, campaign_id, reason, metadata=None):  # pragma: no cover
        raise AssertionError("not used")

    async def mark_send_failed(self, *, campaign_id, error, metadata=None):  # pragma: no cover
        raise AssertionError("not used")

    async def record_webhook_event(self, event):  # pragma: no cover - protocol filler
        raise AssertionError("not used")


class _Audit:
    def __init__(self):
        self.events = []

    async def record(self, event_type, *, campaign_id=None, sequence_id=None, metadata=None):
        self.events.append({
            "event_type": event_type,
            "campaign_id": campaign_id,
            "sequence_id": sequence_id,
            "metadata": metadata,
        })


class _Visibility:
    def __init__(self):
        self.events = []

    async def emit(self, event_type, payload):
        self.events.append({"event_type": event_type, "payload": payload})


class _FailingObservers:
    async def record(self, event_type, *, campaign_id=None, sequence_id=None, metadata=None):
        raise RuntimeError("audit down")

    async def emit(self, event_type, payload):
        raise RuntimeError("visibility down")


@pytest.mark.asyncio
async def test_refresh_calls_repository_and_records_success():
    repo = _CampaignRepo()
    audit = _Audit()
    visibility = _Visibility()
    service = CampaignAnalyticsRefreshService(
        campaigns=repo,
        audit=audit,
        visibility=visibility,
    )

    result = await service.refresh()

    assert result.as_dict() == {"refreshed": True, "error": None}
    assert repo.refresh_calls == 1
    assert audit.events == [{
        "event_type": "analytics_refreshed",
        "campaign_id": None,
        "sequence_id": None,
        "metadata": {"refreshed": True, "error": None},
    }]
    assert visibility.events == [{
        "event_type": "analytics_refreshed",
        "payload": {"refreshed": True, "error": None},
    }]


@pytest.mark.asyncio
async def test_refresh_returns_error_result_without_raising():
    repo = _CampaignRepo(error=RuntimeError("view locked"))
    audit = _Audit()
    service = CampaignAnalyticsRefreshService(campaigns=repo, audit=audit)

    result = await service.refresh()

    assert result.as_dict() == {"refreshed": False, "error": "view locked"}
    assert repo.refresh_calls == 1
    assert audit.events == [{
        "event_type": "analytics_refresh_failed",
        "campaign_id": None,
        "sequence_id": None,
        "metadata": {"refreshed": False, "error": "view locked"},
    }]


@pytest.mark.asyncio
async def test_refresh_does_not_require_observers():
    repo = _CampaignRepo()
    service = CampaignAnalyticsRefreshService(campaigns=repo)

    result = await service.refresh()

    assert result.refreshed is True
    assert repo.refresh_calls == 1


@pytest.mark.asyncio
async def test_refresh_does_not_raise_when_optional_observers_fail():
    repo = _CampaignRepo()
    observers = _FailingObservers()
    service = CampaignAnalyticsRefreshService(
        campaigns=repo,
        audit=observers,
        visibility=observers,
    )

    result = await service.refresh()

    assert result.as_dict() == {"refreshed": True, "error": None}
    assert repo.refresh_calls == 1
