"""The autonomous payment-reminder task must not send, by any route.

ATLAS #2270 / #2271. On 2026-08-03 this task emailed 17 unauthorised dunning
messages to real customers. Every gate that was supposed to hold it closed
defaulted OPEN: the config field shipped ``default=True``, the scheduler seed
omitted ``enabled`` (so ``.get("enabled", True)`` registered the cron), and the
task is not ``enabled_config_key``-managed so the boot sync never reconciles
it. The only thing preventing a send was one line in a hand-maintained ``.env``.

These tests pin the code-level floor that replaced that arrangement. They fail
if someone re-opens any single layer without doing #2270 + #2271 first.
"""

import asyncio
from datetime import date
from unittest.mock import patch

import pytest

from atlas_brain.autonomous.tasks import invoice_payment_reminders as task_mod


def test_autopilot_disabled_flag_is_set():
    """The kill constant is the contract; nothing may quietly clear it."""
    assert task_mod._AUTOPILOT_DISABLED is True


def test_run_returns_before_reading_config_or_invoices():
    """The guard precedes the config read, the overdue query, and any send.

    Patching all three to explode proves the early return is genuinely first:
    if the guard ever moves below them, this test raises instead of passing.
    """

    def _boom(*args, **kwargs):
        raise AssertionError("reminder path executed while autopilot disabled")

    with patch(
        "atlas_brain.storage.repositories.invoice.get_invoice_repo", _boom
    ), patch("atlas_brain.services.email_provider.get_email_provider", _boom):
        result = asyncio.run(task_mod.run(task=None))

    assert "_skip_synthesis" in result
    assert "disabled in code" in result["_skip_synthesis"]


def test_enabling_config_does_not_defeat_the_guard():
    """``ATLAS_INVOICING_REMINDERS_ENABLED=true`` must not produce a send.

    This is the specific accident the constant exists to stop: a config edit,
    or a deploy whose .env lacks the false line, restoring autonomous dunning.
    """

    def _boom(*args, **kwargs):
        raise AssertionError("reminder path executed via config re-enable")

    with patch("atlas_brain.config.settings") as mock_settings, patch(
        "atlas_brain.storage.repositories.invoice.get_invoice_repo", _boom
    ), patch("atlas_brain.services.email_provider.get_email_provider", _boom):
        mock_settings.invoicing.enabled = True
        mock_settings.invoicing.reminders_enabled = True
        result = asyncio.run(task_mod.run(task=None))

    assert "_skip_synthesis" in result


def test_config_default_is_fail_closed():
    """An absent env value means OFF, not ON."""
    from atlas_brain.config import InvoicingConfig

    assert InvoicingConfig.model_fields["reminders_enabled"].default is False


def test_blank_env_value_means_disabled_not_a_crash(monkeypatch):
    """``ATLAS_INVOICING_REMINDERS_ENABLED=`` must be OFF, not ValidationError.

    Pydantic's bool parser rejects ``""``, so before the coercing validator an
    env template rendering the key with an empty value took the whole app down
    at import — and the fail-closed claim on the field was false for exactly
    the shape a half-configured deployment produces.
    """
    from atlas_brain.config import InvoicingConfig

    monkeypatch.setenv("ATLAS_INVOICING_REMINDERS_ENABLED", "")
    assert InvoicingConfig(_env_file=None).reminders_enabled is False

    # Whitespace-only is the same class of half-configured value.
    monkeypatch.setenv("ATLAS_INVOICING_REMINDERS_ENABLED", "   ")
    assert InvoicingConfig(_env_file=None).reminders_enabled is False

    # A real value still parses — the coercion must not swallow an explicit ON.
    monkeypatch.setenv("ATLAS_INVOICING_REMINDERS_ENABLED", "true")
    assert InvoicingConfig(_env_file=None).reminders_enabled is True


def test_clearing_the_guard_restores_the_send(monkeypatch):
    """Permitted side: the constant is what blocks, not something incidental.

    Without this, the suite only proves "nothing sends" — which a broken import
    or an unrelated closed gate would also satisfy. Opening all three gates
    must reach the transport, otherwise the PR's claim that clearing the
    constant restores prior behaviour is unproven.
    """
    from atlas_brain.config import settings

    sent: list[dict] = []

    class FakeEmailProvider:
        async def send(self, *, to, subject, body, attachments=None, **_):
            sent.append({"to": to, "subject": subject})
            return {"ok": True}

    class FakeRepo:
        async def get_overdue(self, as_of_date=None):
            return [{
                "id": "11111111-1111-1111-1111-111111111111",
                "invoice_number": "INV-2026-0001",
                "customer_name": "Guard Probe",
                "customer_email": "probe@example.com",
                "amount_due": 100.0,
                "due_date": date(2026, 1, 1),
                "reminder_count": 0,
                "last_reminder_at": None,
                "contact_id": None,
            }]

        async def update_reminder(self, _id):
            return None

    monkeypatch.setattr(settings.invoicing, "enabled", True)
    monkeypatch.setattr(settings.invoicing, "reminders_enabled", True)
    monkeypatch.setattr(
        "atlas_brain.storage.repositories.invoice.get_invoice_repo",
        lambda: FakeRepo(),
    )
    monkeypatch.setattr(
        "atlas_brain.services.email_provider.get_email_provider",
        lambda: FakeEmailProvider(),
    )
    monkeypatch.setattr(
        "atlas_brain.services.invoice_pdf.render_invoice_pdf",
        lambda inv: b"%PDF-fake",
    )

    # Guard still set, every other gate open -> the guard is the only blocker.
    assert task_mod._AUTOPILOT_DISABLED is True
    blocked = asyncio.run(task_mod.run(task=None))
    assert "_skip_synthesis" in blocked
    assert sent == []

    # Clear only the guard -> the send happens. Same fixtures, one variable.
    monkeypatch.setattr(task_mod, "_AUTOPILOT_DISABLED", False)
    allowed = asyncio.run(task_mod.run(task=None))
    assert allowed.get("reminders_sent") == 1
    assert len(sent) == 1
    assert sent[0]["to"] == ["probe@example.com"]


def test_scheduler_seeds_the_task_disabled():
    """A fresh database must not register the 10:00 cron enabled."""
    from atlas_brain.autonomous.scheduler import TaskScheduler

    seeds = [
        t
        for t in TaskScheduler._DEFAULT_TASKS
        if t["name"] == "invoice_payment_reminders"
    ]
    assert len(seeds) == 1, "expected exactly one seed definition"
    # Explicitly False, not merely absent -- absence is what defaulted True.
    assert seeds[0].get("enabled") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
