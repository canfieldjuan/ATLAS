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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# This contract is a fail-closed security boundary for UNTRUSTED worker output
# (not a trusted-artifact iteration contract), so it is strict: no unmodeled
# fields (extra='forbid'), no type coercion (strict -- "yes"/1 is not True), and
# only the canonical "schema" key on input (no populate_by_name). serialize_by_
# alias still emits the canonical "schema" key on dump.
_BASE_CONFIG = ConfigDict(extra="forbid", strict=True, serialize_by_alias=True)

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

# Closed, ORDERED set of workflow stages (stages_completed / failed_stage). The
# order matters: a failure at stage N means stages before N completed and N and
# later did not.
STAGE_ORDER: tuple[str, ...] = ("lookup", "select", "compose", "approval")
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
    schema_version: Literal[1] = 1
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

    # Normalize-and-store canonical values, so the validator GUARANTEES canonical
    # output downstream (not merely accepts a padded/mixed-case value and keeps the
    # raw one). strings are stripped; approval is also case-folded.
    @field_validator("error_code", "customer_id", "draft_id", mode="before")
    @classmethod
    def _strip_str(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("approval_state", mode="before")
    @classmethod
    def _normalize_approval(cls, value: Any) -> Any:
        return _norm(value) if isinstance(value, str) else value

    @field_validator("next_permitted_actions", mode="before")
    @classmethod
    def _normalize_actions(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [_norm(item) if isinstance(item, str) else item for item in value]
        return value

    # Reject a non-integer schema_version before the Literal check, since
    # Literal[1] would otherwise admit True or 1.0 by equality.
    @field_validator("schema_version", mode="before")
    @classmethod
    def _require_int_version(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value

    @model_validator(mode="after")
    def _fail_closed(self) -> "FollowUpDraftResult":
        if self.success != (self.status == "drafted"):
            raise ValueError("success must be true iff status == 'drafted'")

        if self.status == "drafted":
            if not self.customer_id or not self.draft_id:
                raise ValueError("a drafted result requires non-blank customer_id and draft_id")
            if self.error_code is not None:
                raise ValueError("a drafted result carries no error_code")
            if self.failed_stage is not None:
                raise ValueError("a drafted result may not carry a failed_stage")
        else:
            expected = STATUS_ERROR_CODE[self.status]
            if self.error_code != expected:  # exact canonical (already stripped)
                raise ValueError(
                    f"status {self.status!r} requires canonical error_code {expected!r}"
                )
            if self.failed_stage is None:
                raise ValueError(f"status {self.status!r} requires a failed_stage")
            # Stages at or after the failed stage cannot have completed.
            fail_index = STAGE_ORDER.index(self.failed_stage)
            for stage in self.stages_completed:
                if STAGE_ORDER.index(stage) >= fail_index:
                    raise ValueError(
                        f"stages_completed may not include {stage!r} at or after "
                        f"failed_stage {self.failed_stage!r}"
                    )

        # The worker may not emit "approved" (any case/whitespace); approval is
        # server-owned. Any other value must be a known benign state. approval_state
        # is already normalized (stripped + case-folded) by the field validator.
        if self.approval_state is not None:
            if self.approval_state == "approved":
                raise ValueError(
                    "INVALID_APPROVAL_STATE: the worker may not emit approval_state "
                    "'approved'; approval is server-owned"
                )
            if self.approval_state not in _ALLOWED_APPROVAL:
                raise ValueError(f"unknown approval_state: {self.approval_state!r}")

        # Closed allowlist: any action outside PERMITTED_ACTIONS (any send variant)
        # fails closed.
        for action in self.next_permitted_actions:  # already normalized above
            if action not in PERMITTED_ACTIONS:
                raise ValueError(f"action {action!r} is not a permitted draft-only action")
        return self


def resolve_approval(result: FollowUpDraftResult, *, server_approved: bool) -> bool:
    """Return the authoritative approval for a draft, derived from server-side
    state ONLY -- the worker's ``approval_state`` is never trusted. A result that
    is not a successful ``drafted`` outcome can never be approved."""
    if result.status != "drafted":
        return False
    # Fail closed: approve only on an actual True boolean, never on a truthy
    # serialized value such as "false" or 0 reaching this approval boundary.
    return server_approved is True


def validate_followup_draft(data: dict[str, Any]) -> FollowUpDraftResult:
    """Deterministically validate a raw worker result against the contract,
    failing closed. Raises pydantic ValidationError on any malformed, non-canonical,
    or contradictory result."""
    return FollowUpDraftResult.model_validate(data)
