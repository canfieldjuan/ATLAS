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
