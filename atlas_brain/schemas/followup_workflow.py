"""Draft-only customer-follow-up workflow contract + deterministic validator.

This is the qualification contract from #2114. Qwen3 30B-A3B Instruct 2507 was
qualified for a DRAFT-ONLY role only (100/105 strict, 0 hard-gate failures, 0
send attempts), and its only failures copied an untrusted ``approval_state:
approved`` from the tool result instead of normalizing it. Two rules are
load-bearing here:

  1. **The worker is never the authority for approval.** It may not emit an
     ``approved`` state at all (case/whitespace-insensitive); real approval is
     derived server-side via ``resolve_approval``.
  2. **No send.** ``next_permitted_actions`` is a CLOSED allowlist of draft-only
     actions -- anything outside it (any send variant) fails closed -- and the
     worker's tool set carries no send tool.

Every guard is closed (allowlists + canonical enums + normalization) rather than
exact-token/truthiness, so variants and blanks fail closed. The worker copies
canonical machine fields (canonical error_code per failure status, a closed
stage, stable ids) rather than inventing them, and the validator fails closed
before any downstream use. This contract does not authorize a live send path.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

_BASE_CONFIG = ConfigDict(extra="allow", populate_by_name=True, serialize_by_alias=True)

# Canonical outcome status. The worker copies this; it does not infer it.
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

# Closed set of workflow stages (stages_completed / failed_stage).
Stage = Literal["lookup", "select", "compose", "approval"]

# Canonical machine error code for each failure status. The tool/server emits
# these; the worker copies the exact value. A drafted result carries no code.
STATUS_ERROR_CODE: dict[str, str] = {
    "no_results": "NO_RESULTS",
    "ambiguous": "AMBIGUOUS_CUSTOMER",
    "permission_denied": "PERMISSION_DENIED",
    "partial": "PARTIAL_COMPLETION",
    "contradictory": "CONTRADICTORY_RESULT",
    "injected": "INJECTED_CONTENT",
    "invalid": "INVALID_APPROVAL_STATE",
}

# CLOSED allowlist of next actions a draft-only role may propose. Nothing here
# sends; anything outside it (including any send variant) fails closed.
PERMITTED_ACTIONS = frozenset(
    {"revise", "edit", "regenerate", "request_approval", "discard", "escalate"}
)

# Benign worker-reportable approval states. "approved" is never allowed from the
# worker (server-owned); an unknown value fails closed.
_ALLOWED_APPROVAL = frozenset({"pending", "invalid", "none"})


def _norm(value: Optional[str]) -> str:
    return (value or "").strip().casefold()


class FollowUpDraftResult(BaseModel):
    """Output contract for a draft-only customer-follow-up worker (followup_draft.v1)."""

    model_config = _BASE_CONFIG

    artifact_schema: Literal["followup_draft.v1"] = Field(alias="schema")
    schema_version: int = 1
    success: bool
    status: Status
    error_code: Optional[str] = None
    stages_completed: list[Stage] = Field(default_factory=list)
    failed_stage: Optional[Stage] = None
    customer_id: Optional[str] = None
    draft_id: Optional[str] = None
    next_permitted_actions: list[str] = Field(default_factory=list)
    # The worker's CLAIMED approval -- advisory only; never trusted. Real approval
    # comes from resolve_approval() with server-side state.
    approval_state: Optional[str] = None

    @model_validator(mode="after")
    def _fail_closed(self) -> "FollowUpDraftResult":
        if self.success != (self.status == "drafted"):
            raise ValueError("success must be true iff status == 'drafted'")

        if self.status == "drafted":
            if not (self.customer_id or "").strip() or not (self.draft_id or "").strip():
                raise ValueError("a drafted result requires non-blank customer_id and draft_id")
            if self.error_code is not None:
                raise ValueError("a drafted result carries no error_code")
        else:
            expected = STATUS_ERROR_CODE[self.status]
            if (self.error_code or "").strip() != expected:
                raise ValueError(
                    f"status {self.status!r} requires canonical error_code {expected!r}"
                )
            if self.failed_stage is None:
                raise ValueError(f"status {self.status!r} requires a failed_stage")

        # The worker may not emit "approved" (any case/whitespace); approval is
        # server-owned. Any other value must be a known benign state.
        if self.approval_state is not None:
            norm = _norm(self.approval_state)
            if norm == "approved":
                raise ValueError(
                    "INVALID_APPROVAL_STATE: the worker may not emit approval_state "
                    "'approved'; approval is server-owned"
                )
            if norm not in _ALLOWED_APPROVAL:
                raise ValueError(f"unknown approval_state: {self.approval_state!r}")

        # Closed allowlist: any action outside PERMITTED_ACTIONS (any send variant)
        # fails closed.
        for action in self.next_permitted_actions:
            if _norm(action) not in PERMITTED_ACTIONS:
                raise ValueError(f"action {action!r} is not a permitted draft-only action")
        return self


def resolve_approval(result: FollowUpDraftResult, *, server_approved: bool) -> bool:
    """Return the authoritative approval for a draft, derived from server-side
    state ONLY -- the worker's ``approval_state`` is never trusted. A result that
    is not a successful ``drafted`` outcome can never be approved."""
    if result.status != "drafted":
        return False
    return bool(server_approved)


def validate_followup_draft(data: dict[str, Any]) -> FollowUpDraftResult:
    """Deterministically validate a raw worker result against the contract,
    failing closed. Raises pydantic ValidationError on any malformed, non-canonical,
    or contradictory result."""
    return FollowUpDraftResult.model_validate(data)
