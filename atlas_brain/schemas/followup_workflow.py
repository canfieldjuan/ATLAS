"""Draft-only customer-follow-up workflow contract + deterministic validator.

This is the qualification contract from #2114. Qwen3 30B-A3B Instruct 2507 was
qualified for a DRAFT-ONLY role only (100/105 strict, 0 hard-gate failures, 0
send attempts), and its only failures copied an untrusted ``approval_state:
approved`` from the tool result instead of normalizing it. So two rules are
load-bearing here:

  1. **The worker is never the authority for approval.** ``approval_state`` on
     the worker's output is advisory only; real approval must come from
     authenticated server-side state (see ``resolve_approval``). A claimed
     ``approved`` on a non-drafted result is the corrupted-approval contradiction
     and is rejected.
  2. **No send.** A draft-only result may not propose any send / side-effect
     action, and the worker's tool set carries no send tool. This contract does
     not authorize a live send path.

The worker COPIES canonical machine fields (status, error_code, failed_stage,
stable ids) rather than inferring a normalization, and the validator fails closed
on the qualified failure modes before any downstream use.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

_BASE_CONFIG = ConfigDict(extra="allow", populate_by_name=True, serialize_by_alias=True)

# Canonical outcome status. The worker copies this from the tool result; it does
# not infer it. "drafted" is the only success; the rest map to the seven
# qualified failure groups.
Status = Literal[
    "drafted",
    "no_results",
    "ambiguous",
    "permission_denied",
    "partial",
    "contradictory",
    "injected",
    "invalid",
]

# A draft-only role may never propose any of these as a next action.
SEND_ACTIONS = frozenset(
    {
        "send",
        "send_email",
        "send_sms",
        "send_invoice",
        "send_proposal",
        "send_estimate",
        "send_brand_health_digest",
        "send_test_webhook_tool",
        "approve_and_send",
    }
)


class FollowUpDraftResult(BaseModel):
    """Output contract for a draft-only customer-follow-up worker (followup_draft.v1)."""

    model_config = _BASE_CONFIG

    artifact_schema: Literal["followup_draft.v1"] = Field(alias="schema")
    schema_version: int = 1
    success: bool
    status: Status
    error_code: Optional[str] = None
    stages_completed: list[str] = Field(default_factory=list)
    failed_stage: Optional[str] = None
    customer_id: Optional[str] = None
    draft_id: Optional[str] = None
    next_permitted_actions: list[str] = Field(default_factory=list)
    # The worker's CLAIMED approval -- advisory only. Downstream must derive real
    # approval from server-side state via resolve_approval(), never trust this.
    approval_state: Optional[str] = None

    @model_validator(mode="after")
    def _fail_closed(self) -> "FollowUpDraftResult":
        if self.success != (self.status == "drafted"):
            raise ValueError("success must be true iff status == 'drafted'")
        if self.status != "drafted" and not self.error_code:
            raise ValueError(f"status {self.status!r} requires a machine error_code")
        if self.status != "drafted" and not self.failed_stage:
            raise ValueError(f"status {self.status!r} requires a failed_stage")
        # The worker is never the approval authority: it may not emit an
        # "approved" state at all. In the qualified failure, the worker copied an
        # untrusted approval_state='approved' instead of normalizing it; approval
        # must be derived server-side via resolve_approval(), so any worker-emitted
        # "approved" is rejected as the corrupted-approval failure.
        if self.approval_state == "approved":
            raise ValueError(
                "INVALID_APPROVAL_STATE: the worker may not emit approval_state "
                "'approved'; approval is server-owned"
            )
        proposed_send = SEND_ACTIONS & {
            action.strip().casefold() for action in self.next_permitted_actions
        }
        if proposed_send:
            raise ValueError(
                f"draft-only result may not permit send actions: {sorted(proposed_send)}"
            )
        return self


def resolve_approval(result: FollowUpDraftResult, *, server_approved: bool) -> bool:
    """Return the authoritative approval for a draft, derived from server-side
    state ONLY -- the worker's ``approval_state`` is never trusted. A draft that
    is not a successful ``drafted`` result can never be approved."""
    if result.status != "drafted":
        return False
    return bool(server_approved)


def validate_followup_draft(data: dict[str, Any]) -> FollowUpDraftResult:
    """Deterministically validate a raw worker result against the contract,
    failing closed. Raises pydantic ValidationError (which wraps the fail-closed
    rules) on any malformed or contradictory result."""
    return FollowUpDraftResult.model_validate(data)
