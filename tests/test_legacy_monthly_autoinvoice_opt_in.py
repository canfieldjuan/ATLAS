"""Fail-safe defaults for the legacy monthly auto-invoice task."""

from __future__ import annotations

import builtins
import importlib
from types import SimpleNamespace

import pytest

_LEGACY_WRITEFUL_PROVIDER_MODULES = {
    "atlas_brain.services.calendar_provider",
    "atlas_brain.services.crm_provider",
    "atlas_brain.storage.repositories.customer_service",
    "atlas_brain.storage.repositories.invoice",
}


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
        globals: object = None,
        locals: object = None,
        fromlist: object = (),
        level: int = 0,
    ) -> object:
        if name in _LEGACY_WRITEFUL_PROVIDER_MODULES:
            raise AssertionError(f"disabled task imported writeful provider: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_if_legacy_writer_is_imported)

    result = await task_module.run(SimpleNamespace())

    assert result == {"_skip_synthesis": "Auto-invoicing disabled"}
