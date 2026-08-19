"""Fail-safe defaults for the legacy monthly auto-invoice task."""

from __future__ import annotations

import builtins
import importlib
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
    task_module = importlib.import_module(
        "atlas_brain.autonomous.tasks.monthly_invoice_generation"
    )
    monkeypatch.setattr(config_module.settings.invoicing, "enabled", True)
    monkeypatch.setattr(config_module.settings.invoicing, "auto_invoice_enabled", False)

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

    provider_import_shapes = (
        (
            "services.calendar_provider",
            ("get_calendar_provider",),
            "atlas_brain.services.calendar_provider",
        ),
        (
            "services.crm_provider",
            ("get_crm_provider",),
            "atlas_brain.services.crm_provider",
        ),
        (
            "storage.repositories.customer_service",
            ("get_customer_service_repo",),
            "atlas_brain.storage.repositories.customer_service",
        ),
        (
            "storage.repositories.invoice",
            ("get_invoice_repo",),
            "atlas_brain.storage.repositories.invoice",
        ),
        (
            "services.invoice_pdf",
            ("render_invoice_pdf",),
            "atlas_brain.services.invoice_pdf",
        ),
        (
            "services.email_provider",
            ("get_email_provider",),
            "atlas_brain.services.email_provider",
        ),
        (
            "templates.email.invoice",
            ("BUSINESS_NAME", "BUSINESS_SIGNATURE"),
            "atlas_brain.templates.email.invoice",
        ),
        (
            "tools.notify",
            ("notify_tool",),
            "atlas_brain.tools.notify",
        ),
    )
    for relative_name, fromlist, expected_name in provider_import_shapes:
        with pytest.raises(AssertionError, match=expected_name):
            fail_if_legacy_writer_is_imported(
                relative_name,
                task_module.__dict__,
                None,
                fromlist,
                3,
            )

    monkeypatch.setattr(builtins, "__import__", fail_if_legacy_writer_is_imported)

    result = await task_module.run(SimpleNamespace())

    assert result == {"_skip_synthesis": "Auto-invoicing disabled"}
