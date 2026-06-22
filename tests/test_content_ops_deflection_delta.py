import json

import pytest

from extracted_content_pipeline.deflection_delta import compute_deflection_delta


def _model(*rows: dict[str, object], section_id: str = "backlog_table") -> dict[str, object]:
    return {
        "schema_version": "deflection.v1",
        "title": "Support Ticket Deflection Report",
        "summary": {
            "source_date_start": "2026-05-01",
            "source_date_end": "2026-05-31",
            "source_window_days": 31,
        },
        "sections": [
            {
                "id": section_id,
                "title": "Backlog",
                "priority": 90,
                "surfaces": ["web"],
                "default_limit": 25,
                "required_data": ["items"],
                "snapshot_safe_fields": [],
                "data": {"items": list(rows)},
            }
        ],
    }


def _row(
    key: str,
    *,
    rank: int = 1,
    question: str = "How do I export reports?",
    status: str = "Needs answer",
    ticket_count: int = 2,
    cost: float = 27.0,
    identity_confidence: str = "high",
    csat: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "rank": rank,
        "repeat_key": key,
        "cluster_id": key,
        "identity_basis": "question_topic",
        "identity_confidence": identity_confidence,
        "question": question,
        "status": status,
        "owner_lane": "exports",
        "fix_type": "create_missing_answer",
        "ticket_count": ticket_count,
        "estimated_support_cost": cost,
        "csat_signal": csat
        or {
            "status": "insufficient_data",
            "csat_present_count": 0,
            "negative_csat_ticket_count": 0,
            "numeric_average": None,
        },
    }


def _by_key(delta: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        str(item["identity_key"]): item
        for item in delta["items"]  # type: ignore[index]
    }


def test_deflection_delta_matches_stable_identity_not_rank() -> None:
    baseline = _model(_row("repeat_b001", rank=1, status="Draft ready"))
    current = _model(_row("repeat_b001", rank=7, status="Draft ready"))

    delta = compute_deflection_delta(current, baseline)

    assert json.loads(json.dumps(delta)) == delta
    assert delta["schema_version"] == "deflection_delta.v1"
    assert delta["summary"]["matched_item_count"] == 1  # type: ignore[index]
    assert delta["summary"]["new_count"] == 0  # type: ignore[index]
    assert delta["summary"]["resolved_count"] == 0  # type: ignore[index]
    assert _by_key(delta)["repeat_b001"]["change_types"] == ["STABLE"]


def test_deflection_delta_classifies_new_resolved_and_changed_rows() -> None:
    baseline = _model(
        _row(
            "repeat_b001",
            status="Needs answer",
            ticket_count=3,
            cost=40.5,
            csat={
                "status": "sparse",
                "csat_present_count": 2,
                "negative_csat_ticket_count": 0,
                "numeric_average": None,
            },
        ),
        _row("repeat_b002", question="Where is SSO?", ticket_count=2, cost=27.0),
    )
    current = _model(
        _row(
            "repeat_b001",
            status="Draft ready",
            ticket_count=5,
            cost=67.5,
            csat={
                "status": "present",
                "csat_present_count": 4,
                "negative_csat_ticket_count": 2,
                "numeric_average": 2.0,
            },
        ),
        _row(
            "repeat_b003",
            question="Why did billing come back?",
            status="Already covered but still recurring",
            ticket_count=2,
            cost=27.0,
        ),
    )

    delta = compute_deflection_delta(current, baseline)
    rows = _by_key(delta)

    assert set(rows["repeat_b001"]["change_types"]) == {
        "GROWING",
        "STATUS_CHANGED",
        "COST_CHANGED",
        "CSAT_CHANGED",
    }
    assert rows["repeat_b001"]["ticket_count_delta"] == 2.0
    assert rows["repeat_b001"]["support_cost_delta"] == 27.0
    assert rows["repeat_b002"]["change_types"] == [
        "RESOLVED",
        "SHRINKING",
        "COST_CHANGED",
    ]
    assert rows["repeat_b003"]["change_types"] == [
        "NEW",
        "RESURFACED",
        "GROWING",
        "COST_CHANGED",
    ]
    assert delta["summary"]["new_count"] == 1  # type: ignore[index]
    assert delta["summary"]["resolved_count"] == 1  # type: ignore[index]
    assert delta["summary"]["support_cost_delta"] == 27.0  # type: ignore[index]


def test_deflection_delta_keeps_low_confidence_identity_unmatched() -> None:
    baseline = _model(_row("repeat_b001", identity_confidence="low"))
    current = _model(_row("repeat_b001", identity_confidence="low"))

    delta = compute_deflection_delta(current, baseline)

    assert delta["summary"]["matched_item_count"] == 0  # type: ignore[index]
    assert delta["summary"]["new_count"] == 1  # type: ignore[index]
    assert delta["summary"]["resolved_count"] == 1  # type: ignore[index]
    assert delta["summary"]["low_confidence_identity_count"] == 2  # type: ignore[index]
    assert {item["identity_key"] for item in delta["items"]} == {  # type: ignore[index]
        "unmatched:baseline:1",
        "unmatched:current:1",
    }


def test_deflection_delta_demotes_all_duplicate_identity_rows() -> None:
    baseline = _model(_row("repeat_b001", question="Baseline export question"))
    current = _model(
        _row("repeat_b001", question="First duplicate"),
        _row("repeat_b001", question="Second duplicate"),
    )

    delta = compute_deflection_delta(current, baseline)

    assert delta["summary"]["matched_item_count"] == 0  # type: ignore[index]
    assert delta["summary"]["new_count"] == 2  # type: ignore[index]
    assert delta["summary"]["resolved_count"] == 1  # type: ignore[index]
    assert delta["summary"]["low_confidence_identity_count"] == 3  # type: ignore[index]
    assert {item["identity_key"] for item in delta["items"]} == {  # type: ignore[index]
        "unmatched:baseline:1",
        "unmatched:current:1",
        "unmatched:current:2",
    }
    assert all(  # type: ignore[index]
        "LOW_CONFIDENCE_IDENTITY" in item["change_types"]
        for item in delta["items"]
    )


def test_deflection_delta_counts_missing_identity_as_low_confidence() -> None:
    current = _model(_row("", identity_confidence="high"))
    baseline = _model()

    delta = compute_deflection_delta(current, baseline)

    row = delta["items"][0]  # type: ignore[index]
    assert row["identity_key"] == "unmatched:current:1"
    assert row["identity_confidence"] == "low"
    assert row["change_types"] == [
        "NEW",
        "GROWING",
        "COST_CHANGED",
        "LOW_CONFIDENCE_IDENTITY",
    ]
    assert delta["summary"]["low_confidence_identity_count"] == 1  # type: ignore[index]


def test_deflection_delta_rejects_malformed_numeric_fields() -> None:
    current = _model(_row("repeat_b001", ticket_count="many"))  # type: ignore[arg-type]
    baseline = _model(_row("repeat_b001"))

    with pytest.raises(ValueError, match="invalid deflection delta number"):
        compute_deflection_delta(current, baseline)


def test_deflection_delta_falls_back_to_priority_queue_for_older_models() -> None:
    baseline = _model(_row("repeat_b001", ticket_count=2), section_id="priority_fix_queue")
    current = _model(_row("repeat_b001", ticket_count=3, cost=40.5))

    delta = compute_deflection_delta(current, baseline)

    assert delta["summary"]["matched_item_count"] == 1  # type: ignore[index]
    assert _by_key(delta)["repeat_b001"]["change_types"] == [
        "STILL_UNRESOLVED",
        "GROWING",
        "COST_CHANGED",
    ]


def test_deflection_delta_rejects_unsupported_schema() -> None:
    current = _model(_row("repeat_b001"))
    baseline = _model(_row("repeat_b001"))
    current["schema_version"] = "legacy.v0"

    with pytest.raises(ValueError, match="current_model must use schema_version"):
        compute_deflection_delta(current, baseline)
