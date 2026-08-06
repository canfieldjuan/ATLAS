"""Guard tests for the call-action plan contact-update executor (#2299).

The action plan is LLM-proposed from a call transcript, so its params are
untrusted input arriving at a privileged write. Before this guard,
`_exec_update_contact` forwarded them verbatim and the only constraint was
`DatabaseCRMProvider.update_contact`'s 19-field allow-list -- which permits
`business_context_id` (tenancy) and `source` (provenance).

A guard fails on its second side, so these probe both: allowed fields must
still reach the provider, and forbidden fields must not.
"""

from __future__ import annotations

import logging

import pytest

from atlas_brain.api.comms import call_actions


class _RecordingProvider:
    """Captures what the executor actually hands the provider."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def update_contact(self, contact_id: str, updates: dict):
        self.calls.append((contact_id, dict(updates)))
        return {"id": contact_id, **updates}


@pytest.fixture()
def provider(monkeypatch: pytest.MonkeyPatch) -> _RecordingProvider:
    recorder = _RecordingProvider()
    monkeypatch.setattr(
        "atlas_brain.services.crm_provider.get_crm_provider",
        lambda: recorder,
    )
    return recorder


RECORD = {"contact_id": "11111111-1111-1111-1111-111111111111"}


# ---------------------------------------------------------------------------
# The good side: call-derived fields still work
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_derived_fields_reach_the_provider(provider: _RecordingProvider) -> None:
    result = await call_actions._exec_update_contact(
        RECORD,
        {"email": "a@b.com", "phone": "217-555-0100", "address": "1 Main St"},
    )
    assert len(provider.calls) == 1
    _, updates = provider.calls[0]
    assert updates == {
        "email": "a@b.com",
        "phone": "217-555-0100",
        "address": "1 Main St",
    }
    assert "updated" in result


@pytest.mark.asyncio
async def test_every_allowed_field_is_actually_accepted(provider: _RecordingProvider) -> None:
    """The allow-list must not be narrower than it claims."""
    params = {field: "x" for field in call_actions._PLAN_UPDATABLE_CONTACT_FIELDS}
    await call_actions._exec_update_contact(RECORD, params)
    _, updates = provider.calls[0]
    assert set(updates) == set(call_actions._PLAN_UPDATABLE_CONTACT_FIELDS)


# ---------------------------------------------------------------------------
# The second side: tenancy, provenance and lifecycle cannot be written
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "forbidden",
    [
        "business_context_id",
        "source",
        "source_ref",
        "contact_type",
        "status",
        "lead_stage",
        "lead_owner",
        "next_follow_up_at",
        "tags",
        "id",
    ],
)
@pytest.mark.asyncio
async def test_forbidden_field_never_reaches_the_provider(
    provider: _RecordingProvider, forbidden: str
) -> None:
    await call_actions._exec_update_contact(
        RECORD, {"email": "a@b.com", forbidden: "attacker-supplied"}
    )
    _, updates = provider.calls[0]
    assert forbidden not in updates, f"{forbidden} must not be writable from a plan"
    assert updates == {"email": "a@b.com"}


@pytest.mark.asyncio
async def test_tenancy_only_payload_writes_nothing(provider: _RecordingProvider) -> None:
    """A plan proposing only forbidden fields must not call the provider at all.

    Otherwise the executor would issue an update whose sole effect is bumping
    updated_at, making an attempted tenancy rewrite look like a real edit.
    """
    with pytest.raises(call_actions.PlanActionSkipped):
        await call_actions._exec_update_contact(
            RECORD, {"business_context_id": "churnsignals", "source": "forged"}
        )
    assert provider.calls == []


@pytest.mark.asyncio
async def test_dropped_fields_are_logged_not_silent(
    provider: _RecordingProvider, caplog: pytest.LogCaptureFixture
) -> None:
    """Silently discarding a privileged field is its own failure mode."""
    with caplog.at_level(logging.WARNING, logger="atlas.api.comms.call_actions"):
        await call_actions._exec_update_contact(
            RECORD, {"email": "a@b.com", "business_context_id": "churnsignals"}
        )
    assert any(
        "business_context_id" in record.getMessage() for record in caplog.records
    ), "dropping a privileged field must be visible in the log"


# ---------------------------------------------------------------------------
# Pre-existing behavior that must not change
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_linked_contact_is_still_skipped(provider: _RecordingProvider) -> None:
    result = await call_actions._exec_update_contact({}, {"email": "a@b.com"})
    assert provider.calls == []
    assert result == "Skipped: no linked contact"


@pytest.mark.asyncio
async def test_empty_params_is_still_skipped(provider: _RecordingProvider) -> None:
    result = await call_actions._exec_update_contact(RECORD, {})
    assert provider.calls == []
    assert result == "Skipped: no update params"


# ---------------------------------------------------------------------------
# Class closure over an open key space
# ---------------------------------------------------------------------------

def _arbitrary_keys(count: int) -> list[str]:
    """Deterministic pseudo-arbitrary keys spanning the shapes a producer emits.

    `params` is producer-supplied JSON with an open key space, so enumerating a
    handful of known-bad names proves nothing about the space. These cover
    snake_case, dotted paths, dunders, SQL-ish names, unicode, and near-misses
    of real allow-list members.
    """
    import hashlib

    shapes = [
        "field_{}", "nested.field_{}", "__dunder_{}__", "Field-{}", "  spaced_{}  ",
        "contacts.{}", "email_{}", "{}_id", "ünïcode_{}", "SELECT_{}",
        "business_context_id_{}", "source_{}",
    ]
    keys = []
    for index in range(count):
        digest = hashlib.sha256(str(index).encode()).hexdigest()[:6]
        keys.append(shapes[index % len(shapes)].format(digest))
    return keys


@pytest.mark.asyncio
async def test_arbitrary_unknown_keys_never_reach_the_provider(
    provider: _RecordingProvider,
) -> None:
    """CLOSED / ENUMERATED / default-reject, proven over generated keys."""
    params = {key: "value" for key in _arbitrary_keys(120)}
    await call_actions._exec_update_contact(RECORD, {**params, "email": "a@b.com"})
    _, updates = provider.calls[0]
    assert set(updates) == {"email"}, (
        "only enumerated allow-list members may reach the provider"
    )


@pytest.mark.asyncio
async def test_mixed_payload_keeps_only_allowlist_members(
    provider: _RecordingProvider,
) -> None:
    allowed = sorted(call_actions._PLAN_UPDATABLE_CONTACT_FIELDS)
    params = {field: "v" for field in allowed}
    params.update({key: "v" for key in _arbitrary_keys(60)})
    await call_actions._exec_update_contact(RECORD, params)
    _, updates = provider.calls[0]
    assert set(updates) == set(allowed)


@pytest.mark.asyncio
async def test_only_unknown_keys_raises_skipped_not_success(
    provider: _RecordingProvider,
) -> None:
    """A refused action must not be auditable as executed.

    approve_plan records any non-raising return as status "ok", which counts it
    in `executed`, names it in the CRM interaction summary, persists the plan as
    executed, and lists it under "Completed" in the notification.
    """
    with pytest.raises(call_actions.PlanActionSkipped):
        await call_actions._exec_update_contact(
            RECORD, {key: "v" for key in _arbitrary_keys(10)}
        )
    assert provider.calls == []


@pytest.mark.asyncio
async def test_tenancy_only_payload_raises_skipped(provider: _RecordingProvider) -> None:
    with pytest.raises(call_actions.PlanActionSkipped):
        await call_actions._exec_update_contact(
            RECORD, {"business_context_id": "churnsignals", "source": "forged"}
        )
    assert provider.calls == []
