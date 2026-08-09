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
import sys

import pytest

from atlas_brain.autonomous.tasks import invoice_payment_reminders as task_mod


def test_autopilot_disabled_flag_is_set():
    """The kill constant is the contract; nothing may quietly clear it."""
    assert task_mod._AUTOPILOT_DISABLED is True


def test_run_returns_at_the_autopilot_gate_not_a_later_one():
    """The guard fires FIRST, proven without mocking any first-party code.

    Each gate in ``run`` returns a distinct ``_skip_synthesis`` string, so the
    reason identifies which one fired. If the guard were removed or moved below
    the config read, this same call would return "Invoicing disabled" (or
    "Payment reminders disabled...") instead. Asserting the autopilot reason
    exactly is therefore an ordering proof, and it needs no patching of
    ``get_invoice_repo`` or ``get_email_provider`` -- mocking the edge, not
    internal code (ATLAS #1877).
    """
    result = asyncio.run(task_mod.run(task=None))

    assert result == {"_skip_synthesis": task_mod._AUTOPILOT_DISABLED_REASON}
    assert "disabled in code" in result["_skip_synthesis"]
    # Would be the answer if a later gate had produced it.
    assert result["_skip_synthesis"] != "Invoicing disabled"


def test_config_enabled_does_not_defeat_the_guard(monkeypatch):
    """The headline property: config cannot re-open the send path.

    ``ATLAS_INVOICING_REMINDERS_ENABLED=true`` -- or a deploy whose .env simply
    lacks the false line -- must still produce nothing. The gate above runs with
    invoicing disabled ambiently, so on its own it cannot distinguish "the guard
    blocked" from "the master gate blocked"; this one opens BOTH config gates
    and still requires the autopilot reason, which only the guard can produce.

    Patches the settings object's attributes, not a module-level first-party
    symbol, so it adds no INTERNAL_MOCK against the task (ATLAS #1877).
    """
    from atlas_brain.config import settings

    monkeypatch.setattr(settings.invoicing, "enabled", True)
    monkeypatch.setattr(settings.invoicing, "reminders_enabled", True)

    result = asyncio.run(task_mod.run(task=None))

    assert result == {"_skip_synthesis": task_mod._AUTOPILOT_DISABLED_REASON}
    # With both config gates open, any of these would mean the guard did not fire.
    assert result["_skip_synthesis"] != "Invoicing disabled"
    assert "reminders_enabled=False" not in result["_skip_synthesis"]
    assert "reminders_sent" not in result


def test_run_touches_no_config_repository_or_transport():
    """Fail-on-touch probes for ALL THREE boundaries the guard must precede.

    The earlier version probed only ``invoice_pdf``, so it could not detect the
    guard being moved below the config read or below ``get_invoice_repo`` -- and
    the reason-string test cannot either, because a guard that fires late still
    returns the same reason. That gap is what round 3 flagged.

    ``run`` imports each collaborator INSIDE the function body, so absence from
    ``sys.modules`` after the call is proof the line was never reached. This is a
    touch probe with no mock of any first-party symbol: nothing is patched, so
    it adds no INTERNAL_MOCK against the task (ATLAS #1877) and the maturity
    ratchet stays green. Modules are restored afterwards so the probe cannot
    disturb other tests in the session.
    """
    boundaries = (
        "atlas_brain.config",                        # settings read
        "atlas_brain.storage.repositories.invoice",  # overdue query
        "atlas_brain.services.email_provider",       # transport
        "atlas_brain.services.invoice_pdf",          # attachment render
    )
    saved = {m: sys.modules.get(m) for m in boundaries}
    try:
        for mod in boundaries:
            sys.modules.pop(mod, None)

        result = asyncio.run(task_mod.run(task=None))

        assert result == {"_skip_synthesis": task_mod._AUTOPILOT_DISABLED_REASON}
        for mod in boundaries:
            assert mod not in sys.modules, (
                f"run() reached {mod} before the autopilot guard; the guard no "
                "longer precedes every effectful boundary"
            )
    finally:
        for mod, original in saved.items():
            if original is not None:
                sys.modules[mod] = original
            else:
                sys.modules.pop(mod, None)


def test_guard_is_the_first_statement_of_run():
    """Structural proof that nothing can execute before the guard.

    The touch probes above prove the guard precedes the boundaries that exist
    today. This proves it precedes ANY statement, including one added tomorrow,
    by reading the parsed source: the first statement of ``run`` must be an
    ``if _AUTOPILOT_DISABLED`` whose body returns. A reviewer asked for
    fail-on-touch probes; those were added, but on their own they only cover the
    collaborators someone thought to enumerate.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(task_mod))
    run_fn = next(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run"
    )
    body = list(run_fn.body)
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]                     # skip the docstring

    assert body, "run() has no body"
    first = body[0]
    assert isinstance(first, ast.If), (
        f"the first statement of run() is {type(first).__name__}, not the guard"
    )
    assert isinstance(first.test, ast.Name) and first.test.id == "_AUTOPILOT_DISABLED", (
        "the first statement of run() does not test _AUTOPILOT_DISABLED"
    )
    assert any(isinstance(n, ast.Return) for n in ast.walk(first)), (
        "the guard does not return, so execution would continue past it"
    )


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


def test_garbage_env_value_still_raises(monkeypatch):
    """Second side of the coercion boundary: only BLANK is forgiven.

    The validator exists to stop a blank value crashing startup. It must not
    become a catch-all that silently swallows a typo — `ATLAS_INVOICING_
    REMINDERS_ENABLED=ture` is a misconfiguration the operator needs told
    about, not quietly resolved to a default.
    """
    from pydantic import ValidationError

    from atlas_brain.config import InvoicingConfig

    monkeypatch.setenv("ATLAS_INVOICING_REMINDERS_ENABLED", "ture")
    with pytest.raises(ValidationError):
        InvoicingConfig(_env_file=None)


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
