"""Contract tests for the draft-only customer-follow-up workflow (#2114).

The load-bearing invariants: a worker may never emit an ``approved`` state
(approval is server-owned), a draft-only result may not propose a send, and every
failure carries canonical machine fields. Cases mirror the seven qualified
failure groups.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from atlas_brain.schemas.followup_workflow import (
    FollowUpDraftResult,
    resolve_approval,
    validate_followup_draft,
)

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
        "error_code": "E_" + status.upper(),
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


@pytest.mark.parametrize(
    "status",
    ["no_results", "ambiguous", "permission_denied", "partial", "contradictory", "injected", "invalid"],
)
def test_each_failure_group_validates_with_canonical_fields(status):
    validate_followup_draft(_failure(status))  # ok with error_code + failed_stage


@pytest.mark.parametrize(
    "status",
    ["no_results", "ambiguous", "permission_denied", "partial", "contradictory", "injected", "invalid"],
)
def test_failure_requires_error_code_and_failed_stage(status):
    with pytest.raises(ValidationError):
        validate_followup_draft(_failure(status, error_code=None))
    with pytest.raises(ValidationError):
        validate_followup_draft(_failure(status, failed_stage=None))


def test_success_must_match_drafted():
    with pytest.raises(ValidationError):
        validate_followup_draft({**DRAFTED, "success": False})
    with pytest.raises(ValidationError):
        validate_followup_draft(_failure("invalid", success=True))


def test_worker_may_not_emit_approved_on_failure():
    with pytest.raises(ValidationError):
        validate_followup_draft(
            _failure("invalid", error_code="INVALID_APPROVAL_STATE", approval_state="approved")
        )


def test_worker_may_not_emit_approved_even_on_drafted():
    with pytest.raises(ValidationError):
        validate_followup_draft({**DRAFTED, "approval_state": "approved"})


def test_draft_only_result_may_not_permit_send():
    with pytest.raises(ValidationError):
        validate_followup_draft({**DRAFTED, "next_permitted_actions": ["revise", "send_email"]})
    with pytest.raises(ValidationError):
        validate_followup_draft({**DRAFTED, "next_permitted_actions": ["approve_and_send"]})


def test_unknown_status_rejected():
    with pytest.raises(ValidationError):
        validate_followup_draft(_failure("weird"))


def test_resolve_approval_is_server_owned():
    # A drafted result with a benign worker-supplied approval_state.
    r = FollowUpDraftResult.model_validate({**DRAFTED, "approval_state": "pending"})
    assert resolve_approval(r, server_approved=False) is False
    assert resolve_approval(r, server_approved=True) is True


def test_resolve_approval_never_approves_non_drafted():
    r = validate_followup_draft(_failure("partial", failed_stage="draft"))
    assert resolve_approval(r, server_approved=True) is False
