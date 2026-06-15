"""Unit tests for the Zendesk product-proof corpus sanitizer.

These exercise the pure sanitization path with a canned raw export artifact;
no live Zendesk credentials are required. The round-trip test confirms the
sanitized corpus still drives the real importer's requester-vs-agent split
(the regression that dropping raw author IDs introduced).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from extracted_content_pipeline.support_ticket_zendesk_thread import (
    rows_from_zendesk_full_thread,
)

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "capture_zendesk_product_proof_corpus",
    ROOT / "scripts" / "capture_zendesk_product_proof_corpus.py",
)
assert _SPEC is not None and _SPEC.loader is not None
MOD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = MOD
_SPEC.loader.exec_module(MOD)


RAW_EXPORT = {
    "tickets": [
        {
            "ticket": {
                "id": 101,
                "requester_id": 555,
                "subject": "Charged twice, email jane.doe@acme.com",
                "description": "I was charged twice this month.",
                "status": "solved",
                "satisfaction_rating": {"score": "good", "comment": "thanks jane.doe@acme.com"},
                "submitter_id": 556,
                "organization_id": 999,
                "tags": ["billing", "vip"],
                "custom_fields": [{"id": 1, "value": "acct 12345678"}],
                "url": "https://finetunelab.zendesk.com/api/v2/tickets/101.json",
            },
            "comments": [
                {
                    "public": True,
                    "author_id": 555,
                    "plain_body": "I was charged twice. Call me at 555-123-4567. Acct 12345678.",
                },
                {
                    "public": True,
                    "author_id": 700,
                    "body": "We confirmed the duplicate billing event and refunded the extra $49.",
                },
                {
                    "public": False,
                    "author_id": 700,
                    "body": "Internal note: known proration bug, refund quietly.",
                },
                {
                    # Non-boilerplate automation comment: the importer's auto-ack
                    # text filter would NOT catch this, so the sanitizer must drop
                    # it by role or it would become resolution_text.
                    "public": True,
                    "author_id": 0,
                    "via": {"channel": "rule"},
                    "body": "Ticket reassigned to the tier-2 queue by an automation rule.",
                },
            ],
        },
        {
            "ticket": {
                "id": 102,
                "requester_id": 601,
                "subject": "Refund $49 for 2024 plan",
                "status": "open",
                "satisfaction_rating": "unoffered",
            },
            "comments": [
                {"public": True, "author_id": 601, "body": "Please refund the $49 from my 2024 invoice."},
            ],
        },
    ]
}


def _corpus():
    return MOD.sanitize_zendesk_export(RAW_EXPORT, subdomain="finetunelab", run_tag="t1")


def test_corpus_shape_and_metadata() -> None:
    c = _corpus()
    assert c["source"] == "zendesk_trial_api"
    assert c["subdomain"] == "finetunelab"
    assert c["run_tag"] == "t1"
    assert c["ticket_count"] == 2


def test_local_stable_ids_replace_raw_zendesk_ids() -> None:
    c = _corpus()
    assert [t["id"] for t in c["tickets"]] == ["zd-proof-001", "zd-proof-002"]
    blob = __import__("json").dumps(c)
    # no raw Zendesk numeric identity survives anywhere
    for raw_id in ("101", "102", "555", "556", "601", "700", "999"):
        assert f'"{raw_id}"' not in blob
    assert "finetunelab.zendesk.com/api/v2/tickets" not in blob  # url dropped


def test_roles_are_pseudonymized_not_identity() -> None:
    t0 = _corpus()["tickets"][0]
    assert t0["requester_id"] == "requester"
    roles = [c["author_id"] for c in t0["comments"]]
    # the system/automation comment is dropped, so only requester/agent remain
    assert roles == ["requester", "agent", "agent"]


def test_system_automation_comments_are_excluded() -> None:
    # Regression for the Codex finding: a public automation comment that is NOT
    # auto-ack boilerplate must not survive sanitization (otherwise the importer
    # would emit it as agent resolution_text).
    t0 = _corpus()["tickets"][0]
    assert "system" not in [c["author_id"] for c in t0["comments"]]
    assert all("automation rule" not in c["body"].lower() for c in t0["comments"])
    result = rows_from_zendesk_full_thread(_corpus())
    r1 = {row["ticket_id"]: row for row in result.rows}["zd-proof-001"]
    assert "automation rule" not in r1.get("resolution_text", "").lower()


def test_whitelist_projection_drops_identity_fields() -> None:
    t = _corpus()["tickets"][0]
    assert set(t.keys()) == {
        "id", "requester_id", "subject", "description",
        "status", "satisfaction_rating", "comments", "expected",
    }
    for dropped in ("submitter_id", "organization_id", "tags", "custom_fields", "url"):
        assert dropped not in t
    for comment in t["comments"]:
        assert set(comment.keys()) == {"public", "author_id", "body"}


def test_pii_is_scrubbed_from_text() -> None:
    c = _corpus()
    blob = " ".join(
        [t["subject"] for t in c["tickets"]]
        + [t.get("description", "") for t in c["tickets"]]
        + [cm["body"] for t in c["tickets"] for cm in t["comments"]]
    )
    assert "jane.doe@acme.com" not in blob
    assert "555-123-4567" not in blob
    assert "12345678" not in blob
    assert "[email]" in blob and "[phone]" in blob and "[number]" in blob


def test_short_numbers_survive_scrubbing() -> None:
    t2 = _corpus()["tickets"][1]
    body = " ".join(cm["body"] for cm in t2["comments"])
    assert "$49" in body and "2024" in body


def test_satisfaction_rating_reduced_to_score() -> None:
    tickets = _corpus()["tickets"]
    assert tickets[0]["satisfaction_rating"] == "good"
    assert tickets[1]["satisfaction_rating"] == "unoffered"


def test_private_note_flag_and_public_flags() -> None:
    tickets = _corpus()["tickets"]
    # requester(pub), agent(pub), agent-private; the system comment is dropped
    assert [c["public"] for c in tickets[0]["comments"]] == [True, True, False]
    assert tickets[0]["expected"]["has_private_note"] is True
    assert tickets[1]["expected"]["has_private_note"] is False
    assert tickets[0]["expected"]["cluster_theme"] is None
    assert tickets[0]["expected"]["should_publish_answer"] is None


def test_importer_round_trip_splits_customer_and_agent() -> None:
    # The regression: the sanitized corpus must still let the real importer
    # separate customer wording (description) from agent resolution
    # (resolution_text), instead of collapsing both into resolution_text.
    result = rows_from_zendesk_full_thread(_corpus())
    by_id = {row["ticket_id"]: row for row in result.rows}

    r1 = by_id["zd-proof-001"]
    assert "charged twice" in r1["description"].lower()        # customer wording
    assert "refunded the extra" in r1["resolution_text"].lower()  # agent resolution
    # the split is real: customer text is not in the resolution and vice versa
    assert "refunded the extra" not in r1["description"].lower()
    assert "charged twice" not in r1["resolution_text"].lower()
    # private note excluded; automation/system comment excluded
    assert "proration bug" not in (r1["description"] + r1["resolution_text"]).lower()
    assert "automation rule" not in r1["resolution_text"].lower()

    r2 = by_id["zd-proof-002"]
    assert "refund" in r2["description"].lower()               # requester-only ticket
    assert "resolution_text" not in r2                          # no agent reply


def test_assert_no_secrets_rejects_token_markers() -> None:
    import pytest

    poisoned = {"tickets": [{"authorization": "Basic abc/token:secret"}]}
    with pytest.raises(SystemExit):
        MOD._assert_no_secrets(poisoned)
