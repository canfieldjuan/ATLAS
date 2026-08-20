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
_PROVIDER_IMPORT_PROBES = (
    (
        "from ...services.calendar_provider import get_calendar_provider",
        "atlas_brain.services.calendar_provider",
    ),
    (
        "from ...services import calendar_provider",
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


def _resolve_import_name(name: str, module_globals: object, level: int) -> str:
    """Normalize the relative-import shape passed to ``__import__``."""
    if level == 0 or not isinstance(module_globals, dict):
        return name
    package = module_globals.get("__package__")
    if not isinstance(package, str) or not package:
        return name
    return resolve_name("." * level + name, package)


def _writeful_provider_import_name(
    name: str,
    module_globals: object,
    level: int,
    fromlist: object,
) -> str | None:
    """Return a protected module represented by an import name or child fromlist."""
    resolved_name = _resolve_import_name(name, module_globals, level)
    if resolved_name in _LEGACY_WRITEFUL_PROVIDER_MODULES:
        return resolved_name
    if not isinstance(fromlist, (tuple, list)):
        return None
    for imported_name in fromlist:
        if not isinstance(imported_name, str):
            continue
        child_name = f"{resolved_name}.{imported_name}"
        if child_name in _LEGACY_WRITEFUL_PROVIDER_MODULES:
            return child_name
    return None


def _import_legacy_task_with_writeful_provider_blocker(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, object]:
    """Import the real task after blocking every legacy writer collaborator."""
    config_module = importlib.import_module("atlas_brain.config")
    original_import = builtins.__import__

    def fail_if_legacy_writer_is_imported(
        name: str,
        module_globals: object = None,
        locals: object = None,
        fromlist: object = (),
        level: int = 0,
    ) -> object:
        provider_name = _writeful_provider_import_name(
            name, module_globals, level, fromlist
        )
        if provider_name:
            raise AssertionError(
                f"legacy task imported writeful provider: {provider_name}"
            )
        return original_import(name, module_globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_if_legacy_writer_is_imported)
    # Import the actual task only after the sentinel is live. A future hoisted
    # provider import must fail during module load rather than hiding behind a
    # task result below.
    monkeypatch.delitem(sys.modules, _LEGACY_TASK_MODULE, raising=False)
    task_module = importlib.import_module(_LEGACY_TASK_MODULE)
    assert task_module.__package__ == _LEGACY_TASK_PACKAGE
    return config_module, task_module


def _valid_billing_month_overrides() -> tuple[tuple[str, tuple[int, int]], ...]:
    """Derive edge and ordinary values from the exact calendar-valid grammar."""
    return tuple(
        (f"{year:04d}-{month:02d}", (year, month))
        for year in (1, 2026, 9999)
        for month in (1, 12)
    )


def _malformed_billing_month_overrides() -> tuple[object, ...]:
    """Derive structural and calendar complements of exact ASCII ``YYYY-MM``."""
    year = "2026"
    month = "03"
    return (
        202603,
        "",
        f"{year}{month}",
        f"{year}/{month}",
        f"{year}-{month}-01",
        f"{year}-3",
        f"{year}-003",
        f"{year[:-1]}x-{month}",
        f"{year}-{month[:-1]}x",
        "\uff12\uff10\uff12\uff16-03",
        "0000-03",
        f"{year}-00",
        f"{year}-13",
    )


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
    config_module, task_module = _import_legacy_task_with_writeful_provider_blocker(
        monkeypatch
    )

    for module_scope_import, expected_name in _PROVIDER_IMPORT_PROBES:
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


@pytest.mark.parametrize(
    ("value", "expected"),
    _valid_billing_month_overrides(),
)
def test_legacy_billing_month_parser_admits_exact_calendar_periods(
    value: str,
    expected: tuple[int, int],
) -> None:
    """The one parser recognizes exact ASCII periods across calendar bounds."""
    task_module = importlib.import_module(_LEGACY_TASK_MODULE)

    assert task_module._parse_billing_month_override(value) == expected


@pytest.mark.parametrize("value", _malformed_billing_month_overrides())
def test_legacy_billing_month_parser_rejects_non_evidence_values(value: object) -> None:
    """Structural or calendar evidence missing from an override cannot admit it."""
    task_module = importlib.import_module(_LEGACY_TASK_MODULE)

    assert task_module._parse_billing_month_override(value) is None


@pytest.mark.asyncio
async def test_malformed_legacy_billing_month_returns_before_provider_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled legacy metadata still rejects malformed periods before providers."""
    config_module, task_module = _import_legacy_task_with_writeful_provider_blocker(
        monkeypatch
    )
    monkeypatch.setattr(config_module.settings.invoicing, "enabled", True)
    monkeypatch.setattr(config_module.settings.invoicing, "auto_invoice_enabled", True)

    for value in _malformed_billing_month_overrides():
        result = await task_module.run(
            SimpleNamespace(metadata={"billing_month": value})
        )

        assert result == {
            "_skip_synthesis": (
                f"Invalid billing_month format: {value!r} (expected YYYY-MM)"
            )
        }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata",
    ({}, {"billing_month": None}, {"billing_month": "2026-03"}),
)
async def test_admitted_legacy_billing_month_reaches_provider_boundary(
    monkeypatch: pytest.MonkeyPatch,
    metadata: dict[str, object],
) -> None:
    """Missing/null and valid periods retain their enabled provider admission."""
    config_module, task_module = _import_legacy_task_with_writeful_provider_blocker(
        monkeypatch
    )
    monkeypatch.setattr(config_module.settings.invoicing, "enabled", True)
    monkeypatch.setattr(config_module.settings.invoicing, "auto_invoice_enabled", True)

    with pytest.raises(
        AssertionError,
        match="atlas_brain.services.calendar_provider",
    ):
        await task_module.run(SimpleNamespace(metadata=metadata))
