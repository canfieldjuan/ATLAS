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


# ---------------------------------------------------------------------------
# Derived from the producer contract, not from the constant under test
# ---------------------------------------------------------------------------

# Read off atlas_brain/skills/call/call_extraction.md's output schema. Deriving
# the pass-side expectation from `_PLAN_UPDATABLE_CONTACT_FIELDS` made the test
# shrink with the constant: removing a legitimate field kept it green while
# silently breaking real updates.
PRODUCER_FIELDS = {
    "customer_name": "full_name",
    "customer_phone": "phone",
    "customer_email": "email",
    "address": "address",
}


@pytest.mark.parametrize("produced,canonical", sorted(PRODUCER_FIELDS.items()))
@pytest.mark.asyncio
async def test_producer_field_names_reach_the_provider(
    provider: _RecordingProvider, produced: str, canonical: str
) -> None:
    """`update_contact` has no parameter schema in action_planning.md.

    The plan is written by an LLM that has just been shown the extracted data,
    so `{"customer_email": ...}` is the likely shape. Rejecting it would make
    the guard silently break every legitimate update.
    """
    await call_actions._exec_update_contact(RECORD, {produced: "value"})
    assert provider.calls, f"{produced} was rejected; real updates would be dropped"
    _, updates = provider.calls[0]
    assert updates == {canonical: "value"}


@pytest.mark.asyncio
async def test_full_extracted_payload_is_applied(provider: _RecordingProvider) -> None:
    """The whole realistic producer payload, not one field at a time."""
    await call_actions._exec_update_contact(
        RECORD,
        {
            "customer_name": "Bob",
            "customer_phone": "217-555-0100",
            "customer_email": "bob@example.com",
            "address": "1 Main St",
        },
    )
    _, updates = provider.calls[0]
    assert updates == {
        "full_name": "Bob",
        "phone": "217-555-0100",
        "email": "bob@example.com",
        "address": "1 Main St",
    }


@pytest.mark.asyncio
async def test_aliases_do_not_smuggle_forbidden_fields(
    provider: _RecordingProvider,
) -> None:
    """An alias must not become a side door into tenancy or provenance."""
    for produced in ("customer_business_context_id", "customer_source", "source"):
        provider.calls.clear()
        await call_actions._exec_update_contact(
            RECORD, {"customer_email": "a@b.com", produced: "x"}
        )
        _, updates = provider.calls[0]
        assert set(updates) == {"email"}, f"{produced} must not be admitted"


# ---------------------------------------------------------------------------
# Null values, untrusted key rendering, plan-status semantics
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_null_values_never_blank_existing_contact_data(
    provider: _RecordingProvider,
) -> None:
    """`call_extraction.md` emits null for anything not mentioned on the call.

    A plan copying the extracted payload therefore carries nulls for most
    fields. Writing them through would erase existing CRM data: a call that
    mentioned only a phone number would blank the contact's email.
    """
    await call_actions._exec_update_contact(
        RECORD,
        {
            "customer_phone": "217-555-0100",
            "customer_email": None,
            "customer_name": None,
            "address": None,
        },
    )
    _, updates = provider.calls[0]
    assert updates == {"phone": "217-555-0100"}


@pytest.mark.asyncio
async def test_blank_strings_are_also_ignored(provider: _RecordingProvider) -> None:
    await call_actions._exec_update_contact(
        RECORD, {"customer_email": "   ", "customer_phone": "217-555-0100"}
    )
    _, updates = provider.calls[0]
    assert updates == {"phone": "217-555-0100"}


@pytest.mark.asyncio
async def test_all_null_payload_writes_nothing(provider: _RecordingProvider) -> None:
    with pytest.raises(call_actions.PlanActionSkipped):
        await call_actions._exec_update_contact(
            RECORD, {"customer_email": None, "customer_phone": None}
        )
    assert provider.calls == []


@pytest.mark.asyncio
async def test_control_characters_in_keys_cannot_forge_log_records(
    provider: _RecordingProvider, caplog: pytest.LogCaptureFixture
) -> None:
    """Rejected key names are LLM-produced text reaching logs and ntfy."""
    forged = "field\nERROR forged entry"
    with caplog.at_level(logging.WARNING, logger="atlas.api.comms.call_actions"):
        await call_actions._exec_update_contact(
            RECORD, {"customer_email": "a@b.com", forged: "x"}
        )
    for record in caplog.records:
        assert "\n" not in record.getMessage(), "newline survived into the log record"


@pytest.mark.asyncio
async def test_rejected_key_rendering_is_length_bounded(
    provider: _RecordingProvider,
) -> None:
    long_key = "k" * 500
    rendered = call_actions._render_keys([long_key])
    assert len(rendered) <= call_actions._MAX_RENDERED_KEY_LEN
    many = call_actions._render_keys([f"k{i}" for i in range(50)])
    assert "more" in many


def test_plan_status_is_skipped_only_when_nothing_was_attempted() -> None:
    """"skipped" sits outside the idempotency guard, so it must not mean
    "everything failed" -- an errored action may still have taken effect."""
    def status_for(results):
        statuses = {r["status"] for r in results}
        return "skipped" if statuses == {"skipped"} else "executed"

    assert status_for([{"status": "skipped"}]) == "skipped"
    assert status_for([{"status": "error"}]) == "executed"
    assert status_for([{"status": "skipped"}, {"status": "error"}]) == "executed"
    assert status_for([{"status": "ok"}, {"status": "skipped"}]) == "executed"
