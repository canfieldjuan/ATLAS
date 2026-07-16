"""Contract tests for the draft-only customer-follow-up workflow (#2114).

Every guard is closed: canonical error codes per failure status, a closed stage
enum, a closed next-action allowlist, and normalized approval handling. The
load-bearing invariant is that the worker is never the approval authority.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from atlas_brain.schemas.followup_workflow import (
    STATUS_ERROR_CODE,
    FollowUpDraftResult,
    resolve_approval,
    validate_followup_draft,
)

FAILURE_STATUSES = list(STATUS_ERROR_CODE)

DRAFTED = {
    "schema": "followup_draft.v1",
    "success": True,
    "status": "drafted",
    "customer_id": "c1",
    "draft_id": "d1",
    "next_permitted_actions": ["revise", "request_approval"],
}


def _failure(status, **over):
    base = {
        "schema": "followup_draft.v1",
        "success": False,
        "status": status,
        "error_code": STATUS_ERROR_CODE[status],
        "failed_stage": "lookup",
    }
    base.update(over)
    return base


def test_valid_drafted_result():
    r = validate_followup_draft(DRAFTED)
    assert r.status == "drafted" and r.success is True


def test_canonical_schema_key_and_ids_preserved():
    d = validate_followup_draft(DRAFTED).model_dump()
    assert d["schema"] == "followup_draft.v1"
    assert d["customer_id"] == "c1" and d["draft_id"] == "d1"
    assert "artifact_schema" not in d


# --- failure fields: canonical + non-blank + closed stage ---


@pytest.mark.parametrize("status", FAILURE_STATUSES)
def test_each_failure_group_validates_with_canonical_fields(status):
    validate_followup_draft(_failure(status))


@pytest.mark.parametrize("status", FAILURE_STATUSES)
def test_failure_requires_error_code_and_failed_stage(status):
    with pytest.raises(ValidationError):
        validate_followup_draft(_failure(status, error_code=None))
    with pytest.raises(ValidationError):
        validate_followup_draft(_failure(status, failed_stage=None))


def test_failure_error_code_must_be_canonical():
    with pytest.raises(ValidationError):
        validate_followup_draft(_failure("no_results", error_code="BANANA"))
    with pytest.raises(ValidationError):  # wrong status's canonical code
        validate_followup_draft(_failure("no_results", error_code="PERMISSION_DENIED"))


def test_blank_error_code_rejected():
    with pytest.raises(ValidationError):
        validate_followup_draft(_failure("partial", error_code="   "))


def test_invalid_failed_stage_rejected():
    with pytest.raises(ValidationError):  # "send" is not a workflow stage
        validate_followup_draft(_failure("partial", failed_stage="send"))


def test_drafted_carries_no_error_code():
    with pytest.raises(ValidationError):
        validate_followup_draft({**DRAFTED, "error_code": "NO_RESULTS"})


# --- success / stable ids ---


def test_success_must_match_drafted():
    with pytest.raises(ValidationError):
        validate_followup_draft({**DRAFTED, "success": False})
    with pytest.raises(ValidationError):
        validate_followup_draft(_failure("invalid", success=True))


@pytest.mark.parametrize("drop", ["customer_id", "draft_id"])
def test_drafted_requires_stable_ids(drop):
    with pytest.raises(ValidationError):
        validate_followup_draft({**DRAFTED, drop: None})
    with pytest.raises(ValidationError):
        validate_followup_draft({**DRAFTED, drop: "  "})


# --- approval: worker is never the authority ---


def test_worker_may_not_emit_approved_on_failure():
    with pytest.raises(ValidationError):
        validate_followup_draft(_failure("invalid", approval_state="approved"))


def test_worker_may_not_emit_approved_on_drafted():
    with pytest.raises(ValidationError):
        validate_followup_draft({**DRAFTED, "approval_state": "approved"})


@pytest.mark.parametrize("token", ["Approved", "APPROVED", "  approved  ", "aPProved"])
def test_approved_variants_rejected(token):
    with pytest.raises(ValidationError):
        validate_followup_draft({**DRAFTED, "approval_state": token})


def test_unknown_approval_state_rejected():
    with pytest.raises(ValidationError):
        validate_followup_draft({**DRAFTED, "approval_state": "definitely-approved"})


def test_benign_approval_state_allowed():
    r = validate_followup_draft({**DRAFTED, "approval_state": "pending"})
    assert r.approval_state == "pending"


# --- no send: closed action allowlist ---


@pytest.mark.parametrize(
    "action", ["send_email", "approve_and_send", "send email", "send-email", "send invoice", "delete_customer"]
)
def test_non_permitted_actions_rejected(action):
    with pytest.raises(ValidationError):
        validate_followup_draft({**DRAFTED, "next_permitted_actions": ["revise", action]})


def test_unknown_status_rejected():
    with pytest.raises(ValidationError):
        validate_followup_draft(
            {
                "schema": "followup_draft.v1",
                "success": False,
                "status": "weird",
                "error_code": "X",
                "failed_stage": "lookup",
            }
        )


# --- server-owned approval resolution ---


def test_resolve_approval_is_server_owned():
    r = validate_followup_draft({**DRAFTED, "approval_state": "pending"})
    assert resolve_approval(r, server_approved=False) is False
    assert resolve_approval(r, server_approved=True) is True


def test_resolve_approval_never_approves_non_drafted():
    r = validate_followup_draft(_failure("partial", failed_stage="compose"))
    assert resolve_approval(r, server_approved=True) is False


# --- strict fail-closed input boundary (Codex round 2) ---


def test_resolve_approval_requires_real_true():
    r = validate_followup_draft(DRAFTED)
    assert resolve_approval(r, server_approved=True) is True
    # a truthy serialized value at the approval boundary must not approve
    assert resolve_approval(r, server_approved="false") is False
    assert resolve_approval(r, server_approved=1) is False


def test_extra_worker_fields_rejected():
    with pytest.raises(ValidationError):
        validate_followup_draft({**DRAFTED, "send_payload": "x@example.com"})


def test_non_v1_schema_version_rejected():
    with pytest.raises(ValidationError):
        validate_followup_draft({**DRAFTED, "schema_version": 2})


def test_attribute_name_only_schema_key_rejected():
    payload = {k: v for k, v in DRAFTED.items() if k != "schema"}
    payload["artifact_schema"] = "followup_draft.v1"
    with pytest.raises(ValidationError):
        validate_followup_draft(payload)


@pytest.mark.parametrize("bad", ["yes", "true", 1])
def test_success_must_be_a_real_boolean(bad):
    with pytest.raises(ValidationError):
        validate_followup_draft({**DRAFTED, "success": bad})


# --- canonical-stored values + remaining strictness (Codex round 3) ---


def test_padded_error_code_is_stored_canonical():
    r = validate_followup_draft(_failure("no_results", error_code="  NO_RESULTS  "))
    assert r.error_code == "NO_RESULTS"  # normalized, not the padded raw value


def test_padded_approval_is_stored_canonical():
    r = validate_followup_draft({**DRAFTED, "approval_state": "  Pending  "})
    assert r.approval_state == "pending"


@pytest.mark.parametrize("bad", [True, 1.0, "1"])
def test_schema_version_must_be_a_real_integer(bad):
    with pytest.raises(ValidationError):
        validate_followup_draft({**DRAFTED, "schema_version": bad})


def test_drafted_may_not_carry_failed_stage():
    with pytest.raises(ValidationError):
        validate_followup_draft({**DRAFTED, "failed_stage": "lookup"})
