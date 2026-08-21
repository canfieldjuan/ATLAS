"""Fail-safe admission proofs for the legacy monthly auto-invoice task."""

from __future__ import annotations

import ast
import builtins
import importlib
import re
import sys
from calendar import monthrange
from datetime import date, datetime, timedelta, timezone
from importlib.util import find_spec, resolve_name
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest

_LEGACY_TASK_MODULE = "atlas_brain.autonomous.tasks.monthly_invoice_generation"
_LEGACY_TASK_PACKAGE = _LEGACY_TASK_MODULE.rpartition(".")[0]
_TASK_LOCAL_SAFE_IMPORT_SUFFIX = ".config"
_STRUCTURAL_SYMBOLS = ("0", "x", "-", "\uff12", "\u00e9")


def _resolve_import_name(name: str, module_globals: object, level: int) -> str:
    """Normalize the relative-import shape passed to ``__import__``."""
    if level == 0 or not isinstance(module_globals, dict):
        return name
    package = module_globals.get("__package__")
    if not isinstance(package, str) or not package:
        return name
    return resolve_name("." * level + name, package)


def _task_local_provider_imports() -> tuple[tuple[str, str], ...]:
    """Derive every non-config task-function import from the real task AST."""
    spec = find_spec(_LEGACY_TASK_MODULE)
    assert spec is not None and spec.origin is not None
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    function_nodes = tuple(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    assert function_nodes

    imports: list[tuple[str, str]] = []
    module_globals = {"__package__": _LEGACY_TASK_PACKAGE}
    for function_node in function_nodes:
        for node in ast.walk(function_node):
            if not isinstance(node, ast.ImportFrom):
                continue
            import_target = _resolve_import_name(
                node.module or "", module_globals, node.level
            )
            if import_target.endswith(_TASK_LOCAL_SAFE_IMPORT_SUFFIX):
                continue
            imports.append((ast.unparse(node), import_target))

    derived_imports = tuple(dict.fromkeys(imports))
    assert derived_imports
    return derived_imports


def _writeful_provider_import_name(
    name: str,
    module_globals: object,
    level: int,
    fromlist: object,
    protected_modules: frozenset[str],
) -> str | None:
    """Return a task-derived protected module represented by this import."""
    resolved_name = _resolve_import_name(name, module_globals, level)
    if resolved_name in protected_modules:
        return resolved_name
    if not isinstance(fromlist, (tuple, list)):
        return None
    for imported_name in fromlist:
        if not isinstance(imported_name, str):
            continue
        child_name = f"{resolved_name}.{imported_name}"
        if child_name in protected_modules:
            return child_name
    return None


def _import_legacy_task_with_writeful_provider_blocker(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, object, tuple[tuple[str, str], ...]]:
    """Import the real task after blocking every task-derived collaborator."""
    config_module = importlib.import_module("atlas_brain.config")
    provider_imports = _task_local_provider_imports()
    protected_modules = frozenset(target for _, target in provider_imports)
    original_import = builtins.__import__

    def fail_if_legacy_writer_is_imported(
        name: str,
        module_globals: object = None,
        locals: object = None,
        fromlist: object = (),
        level: int = 0,
    ) -> object:
        provider_name = _writeful_provider_import_name(
            name,
            module_globals,
            level,
            fromlist,
            protected_modules,
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
    return config_module, task_module, provider_imports


def _billing_month_oracle(value: object) -> tuple[int, int] | None:
    """Independent exact-grammar/calendar/executable-domain parser oracle."""
    if not isinstance(value, str) or re.fullmatch(r"[0-9]{4}-[0-9]{2}", value) is None:
        return None
    year = int(value[:4])
    month = int(value[5:])
    if not (date.min.year <= year <= date.max.year and 1 <= month <= 12):
        return None
    last_day = monthrange(year, month)[1]
    range_start = datetime(year, month, last_day, tzinfo=timezone.utc)
    latest_range_start = datetime.max.replace(tzinfo=timezone.utc) - timedelta(hours=30)
    if range_start > latest_range_start:
        return None
    return year, month


def _all_exact_billing_month_grammar_values():
    """Generate every fixed-width ASCII numeric wire form before semantics."""
    for year in range(10_000):
        for month in range(100):
            yield f"{year:04d}-{month:02d}"


def _structural_billing_month_equivalence_values():
    """Generate every short form over parser-relevant character classes."""
    for length in range(8):
        for characters in product(_STRUCTURAL_SYMBOLS, repeat=length):
            yield "".join(characters)
    for length in (8, 9, 64):
        for symbol in _STRUCTURAL_SYMBOLS:
            yield symbol * length
    yield from (None, 202603, b"2026-03", (), {})


def _calendar_valid_but_nonexecutable_billing_month_values():
    """Derive every valid month whose existing range end cannot be represented."""
    latest_range_start = datetime.max.replace(tzinfo=timezone.utc) - timedelta(hours=30)
    for year in range(date.min.year, date.max.year + 1):
        for month in range(1, 13):
            last_day = monthrange(year, month)[1]
            range_start = datetime(year, month, last_day, tzinfo=timezone.utc)
            if range_start > latest_range_start:
                yield f"{year:04d}-{month:02d}"


def test_calendar_valid_nonexecutable_billing_month_class_is_nonempty() -> None:
    """The real-entrypoint rejection basis includes an executable-range complement."""
    values = tuple(_calendar_valid_but_nonexecutable_billing_month_values())

    assert values
    for value in values:
        assert re.fullmatch(r"[0-9]{4}-[0-9]{2}", value) is not None
        year = int(value[:4])
        month = int(value[5:])
        assert date.min.year <= year <= date.max.year
        assert 1 <= month <= 12
        assert _billing_month_oracle(value) is None


def _provider_boundary_rejection_probes() -> tuple[object, ...]:
    """Derive pre-provider rejections from grammar and executable complements."""
    accepted = "2026-03"
    probes: list[object] = [202603, "", accepted[:-1], accepted + "0"]
    for index, expected_character in enumerate(accepted):
        for replacement in _STRUCTURAL_SYMBOLS:
            if replacement == expected_character:
                continue
            candidate = accepted[:index] + replacement + accepted[index + 1 :]
            if _billing_month_oracle(candidate) is None:
                probes.append(candidate)
    probes.extend(("0000-01", "2026-00", "2026-13"))
    probes.extend(_calendar_valid_but_nonexecutable_billing_month_values())
    return tuple(dict.fromkeys(probes))


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
@pytest.mark.parametrize(
    ("invoicing_enabled", "auto_invoice_enabled", "expected_skip"),
    (
        (False, True, "Invoicing disabled"),
        (True, False, "Auto-invoicing disabled"),
    ),
)
async def test_disabled_legacy_task_returns_before_writeful_provider_imports(
    monkeypatch: pytest.MonkeyPatch,
    invoicing_enabled: bool,
    auto_invoice_enabled: bool,
    expected_skip: str,
) -> None:
    """Each real feature-gate exit precedes every task-derived collaborator."""
    config_module, task_module, provider_imports = (
        _import_legacy_task_with_writeful_provider_blocker(monkeypatch)
    )

    for module_scope_import, expected_name in provider_imports:
        module_scope_globals = {
            "__name__": f"{_LEGACY_TASK_MODULE}_provider_import_probe",
            "__package__": _LEGACY_TASK_PACKAGE,
        }
        with pytest.raises(AssertionError, match=re.escape(expected_name)):
            exec(module_scope_import, module_scope_globals)

    monkeypatch.setattr(config_module.settings.invoicing, "enabled", invoicing_enabled)
    monkeypatch.setattr(
        config_module.settings.invoicing,
        "auto_invoice_enabled",
        auto_invoice_enabled,
    )

    result = await task_module.run(SimpleNamespace())

    assert result == {"_skip_synthesis": expected_skip}


def test_legacy_billing_month_parser_matches_every_exact_numeric_wire_form() -> None:
    """Every exact numeric wire form agrees with the independent oracle."""
    task_module = importlib.import_module(_LEGACY_TASK_MODULE)

    for value in _all_exact_billing_month_grammar_values():
        assert task_module._parse_billing_month_override(
            value
        ) == _billing_month_oracle(value)


def test_legacy_billing_month_parser_matches_structural_equivalence_classes() -> None:
    """Non-string, length, character-class, and separator forms fail closed."""
    task_module = importlib.import_module(_LEGACY_TASK_MODULE)

    for value in _structural_billing_month_equivalence_values():
        assert task_module._parse_billing_month_override(
            value
        ) == _billing_month_oracle(value)


@pytest.mark.asyncio
async def test_malformed_legacy_billing_month_returns_before_provider_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled legacy metadata still rejects the grammar complement pre-provider."""
    config_module, task_module, _ = _import_legacy_task_with_writeful_provider_blocker(
        monkeypatch
    )
    monkeypatch.setattr(config_module.settings.invoicing, "enabled", True)
    monkeypatch.setattr(config_module.settings.invoicing, "auto_invoice_enabled", True)

    for value in _provider_boundary_rejection_probes():
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
    config_module, task_module, _ = _import_legacy_task_with_writeful_provider_blocker(
        monkeypatch
    )
    monkeypatch.setattr(config_module.settings.invoicing, "enabled", True)
    monkeypatch.setattr(config_module.settings.invoicing, "auto_invoice_enabled", True)

    with pytest.raises(
        AssertionError,
        match="atlas_brain.services.calendar_provider",
    ):
        await task_module.run(SimpleNamespace(metadata=metadata))
