"""Fail-safe defaults for the legacy monthly auto-invoice task."""

from __future__ import annotations

import builtins
import importlib
import sys
from importlib.util import resolve_name
from types import SimpleNamespace

import pytest

_LEGACY_WRITEFUL_PROVIDER_MODULES = {
    "atlas_brain.services.calendar_provider",
    "atlas_brain.services.crm_provider",
    "atlas_brain.storage.repositories.customer_service",
    "atlas_brain.storage.repositories.invoice",
    "atlas_brain.services.invoice_pdf",
    "atlas_brain.services.email_provider",
    "atlas_brain.templates.email.invoice",
    "atlas_brain.tools.notify",
}
_LEGACY_TASK_MODULE = "atlas_brain.autonomous.tasks.monthly_invoice_generation"
_LEGACY_TASK_PACKAGE = _LEGACY_TASK_MODULE.rpartition(".")[0]


def _resolve_import_name(name: str, module_globals: object, level: int) -> str:
    """Normalize the relative-import shape passed to ``__import__``."""
    if level == 0 or not isinstance(module_globals, dict):
        return name
    package = module_globals.get("__package__")
    if not isinstance(package, str) or not package:
        return name
    return resolve_name("." * level + name, package)


def test_legacy_monthly_automatic_write_flags_default_off_without_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An absent deployment setting cannot enable the legacy writer or sender."""
    from atlas_brain.config import InvoicingConfig

    monkeypatch.delenv("ATLAS_INVOICING_AUTO_INVOICE_ENABLED", raising=False)
    monkeypatch.delenv("ATLAS_INVOICING_AUTO_INVOICE_SEND_EMAIL", raising=False)

    config = InvoicingConfig(_env_file=None)

    assert config.auto_invoice_enabled is False
    assert config.auto_invoice_send_email is False
    assert config.auto_invoice_review_mode is True


def test_legacy_monthly_automatic_write_flags_keep_explicit_opt_in() -> None:
    """An intentionally operated legacy task can still opt into both flags."""
    from atlas_brain.config import InvoicingConfig

    config = InvoicingConfig(
        _env_file=None,
        auto_invoice_enabled=True,
        auto_invoice_send_email=True,
    )

    assert config.auto_invoice_enabled is True
    assert config.auto_invoice_send_email is True


@pytest.mark.asyncio
async def test_disabled_legacy_task_returns_before_writeful_provider_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real task exit happens before it can reach financial collaborators."""
    config_module = importlib.import_module("atlas_brain.config")
    original_import = builtins.__import__

    def fail_if_legacy_writer_is_imported(
        name: str,
        module_globals: object = None,
        locals: object = None,
        fromlist: object = (),
        level: int = 0,
    ) -> object:
        resolved_name = _resolve_import_name(name, module_globals, level)
        if resolved_name in _LEGACY_WRITEFUL_PROVIDER_MODULES:
            raise AssertionError(
                f"disabled task imported writeful provider: {resolved_name}"
            )
        return original_import(name, module_globals, locals, fromlist, level)

    provider_import_probes = (
        (
            "from ...services.calendar_provider import get_calendar_provider",
            "atlas_brain.services.calendar_provider",
        ),
        (
            "from ...services.crm_provider import get_crm_provider",
            "atlas_brain.services.crm_provider",
        ),
        (
            "from ...storage.repositories.customer_service import get_customer_service_repo",
            "atlas_brain.storage.repositories.customer_service",
        ),
        (
            "from ...storage.repositories.invoice import get_invoice_repo",
            "atlas_brain.storage.repositories.invoice",
        ),
        (
            "from ...services.invoice_pdf import render_invoice_pdf",
            "atlas_brain.services.invoice_pdf",
        ),
        (
            "from ...services.email_provider import get_email_provider",
            "atlas_brain.services.email_provider",
        ),
        (
            "from ...templates.email.invoice import BUSINESS_NAME, BUSINESS_SIGNATURE",
            "atlas_brain.templates.email.invoice",
        ),
        (
            "from ...tools.notify import notify_tool",
            "atlas_brain.tools.notify",
        ),
    )
    monkeypatch.setattr(builtins, "__import__", fail_if_legacy_writer_is_imported)
    # Import the actual task only after the sentinel is live. A future hoisted
    # provider import must fail during module load rather than hiding behind the
    # disabled run() result below.
    monkeypatch.delitem(sys.modules, _LEGACY_TASK_MODULE, raising=False)
    task_module = importlib.import_module(_LEGACY_TASK_MODULE)
    assert task_module.__package__ == _LEGACY_TASK_PACKAGE

    for module_scope_import, expected_name in provider_import_probes:
        module_scope_globals = {
            "__name__": f"{_LEGACY_TASK_MODULE}_provider_import_probe",
            "__package__": _LEGACY_TASK_PACKAGE,
        }
        with pytest.raises(AssertionError, match=expected_name):
            exec(module_scope_import, module_scope_globals)

    monkeypatch.setattr(config_module.settings.invoicing, "enabled", True)
    monkeypatch.setattr(config_module.settings.invoicing, "auto_invoice_enabled", False)

    result = await task_module.run(SimpleNamespace())

    assert result == {"_skip_synthesis": "Auto-invoicing disabled"}
