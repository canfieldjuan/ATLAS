from __future__ import annotations

import pytest

from extracted_content_pipeline.support_ticket_context_contract import (
    SUPPORT_TICKET_SOURCE,
    SUPPORT_TICKET_TOPIC_TYPE,
    UPLOADED_SUPPORT_TICKETS_SOURCE_PERIOD,
    UPLOADED_TICKETS_REVIEW_PERIOD,
    is_support_ticket_context,
    is_support_ticket_topic_type,
    is_uploaded_ticket_context,
    support_ticket_topic_filter,
)


def test_support_ticket_topic_filter_uses_canonical_topic_type() -> None:
    assert support_ticket_topic_filter() == {"topic_type": SUPPORT_TICKET_TOPIC_TYPE}
    assert is_support_ticket_topic_type(SUPPORT_TICKET_TOPIC_TYPE) is True
    assert is_support_ticket_topic_type(f" {SUPPORT_TICKET_TOPIC_TYPE} ") is True


@pytest.mark.parametrize(
    "value",
        [
            "",
            "content_ops_live_smoke",
            None,
        ],
)
def test_support_ticket_topic_type_rejects_non_canonical_values(value: object) -> None:
    assert is_support_ticket_topic_type(value) is False


@pytest.mark.parametrize(
    "context",
    [
        {"source_period": UPLOADED_SUPPORT_TICKETS_SOURCE_PERIOD},
        {"review_period": UPLOADED_TICKETS_REVIEW_PERIOD},
    ],
)
def test_uploaded_ticket_context_detection_markers_bite(
    context: dict[str, object],
) -> None:
    assert is_uploaded_ticket_context(context) is True


@pytest.mark.parametrize(
    "context",
    [
        {},
        {"source_period": "Last 90 days of support tickets"},
        {"review_period": "last 90 days"},
        {"source_period": "Uploaded billing tickets"},
        {"review_period": "uploaded reviews"},
    ],
)
def test_uploaded_ticket_context_detection_rejects_false_positives(
    context: dict[str, object],
) -> None:
    assert is_uploaded_ticket_context(context) is False


@pytest.mark.parametrize(
    "context",
    [
        {"source": SUPPORT_TICKET_SOURCE},
        {"provider": "support_ticket_upload"},
        {"source_period": "Last 90 days of support tickets"},
        {"source_period": "support-ticket upload"},
        {"category": "support tickets"},
        {"topic": "support ticket FAQ gaps"},
        {"included_ticket_row_count": 3},
        {"question_like_ticket_count": 2},
        {"top_ticket_clusters": [{"label": "account", "count": 2}]},
        {"top_clusters": [{"label": "reporting", "count": 1}]},
    ],
)
def test_support_ticket_context_detection_markers_bite(
    context: dict[str, object],
) -> None:
    assert is_support_ticket_context(context) is True


@pytest.mark.parametrize(
    "context",
    [
        {},
        {"source": "review_provider"},
        {"provider": "ticketing_system"},
        {"top_clusters": "pricing, onboarding"},
        {"top_ticket_clusters": "account"},
        {"top_clusters": []},
        {"top_ticket_clusters": []},
        {"included_ticket_row_count": 0},
        {"question_like_ticket_count": ""},
    ],
)
def test_support_ticket_context_detection_rejects_false_positives(
    context: dict[str, object],
) -> None:
    assert is_support_ticket_context(context) is False
