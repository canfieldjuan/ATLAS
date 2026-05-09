"""Standalone-mode regression test for the competitive_intelligence task.

Pins the contract that
``extracted_content_pipeline.autonomous.tasks.competitive_intelligence``
imports cleanly under ``EXTRACTED_PIPELINE_STANDALONE=1``. Pre-PR the
mirror failed with ``ModuleNotFoundError: No module named
'extracted_content_pipeline.services.brand_registry'``; the standalone
substrate added in PR-Decouple-CompIntel-1 satisfies that import.

Standalone contract is import-only -- the task has 24 real DB calls
the fake ``_StandalonePool`` cannot service. Runtime end-to-end
runnability is out of scope; see plans/PR-Decouple-CompIntel-1.md.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from typing import Iterator

import pytest


_STANDALONE_ENV_VAR = "EXTRACTED_PIPELINE_STANDALONE"

_AFFECTED_MODULES = (
    "extracted_content_pipeline.services.brand_registry",
    "extracted_content_pipeline.autonomous.tasks.competitive_intelligence",
)


def _evict(*module_names: str) -> None:
    for name in module_names:
        sys.modules.pop(name, None)


@pytest.fixture
def standalone_brand_registry(monkeypatch) -> Iterator[object]:
    """Force a fresh import of the brand_registry shim under
    ``EXTRACTED_PIPELINE_STANDALONE=1``. Yields the imported module."""
    monkeypatch.setenv(_STANDALONE_ENV_VAR, "1")
    _evict(*_AFFECTED_MODULES)
    try:
        module = importlib.import_module(
            "extracted_content_pipeline.services.brand_registry"
        )
        yield module
    finally:
        _evict(*_AFFECTED_MODULES)


def test_brand_registry_shim_exports_required_symbols(standalone_brand_registry):
    module = standalone_brand_registry
    assert hasattr(module, "resolve_brand_name_cached")
    assert hasattr(module, "_ensure_cache")


def test_resolve_brand_name_cached_is_identity_passthrough(standalone_brand_registry):
    resolve = standalone_brand_registry.resolve_brand_name_cached
    assert resolve("Acme Corp") == "Acme Corp"
    assert resolve(None) is None
    assert resolve("") == ""


def test_ensure_cache_is_async_noop(standalone_brand_registry):
    result = asyncio.run(standalone_brand_registry._ensure_cache())
    assert result is None


def test_competitive_intelligence_mirror_imports_under_standalone(monkeypatch):
    """The load-bearing assertion for this PR.

    Pre-PR this raised ``ModuleNotFoundError: No module named
    'extracted_content_pipeline.services.brand_registry'``.
    """
    monkeypatch.setenv(_STANDALONE_ENV_VAR, "1")
    _evict(*_AFFECTED_MODULES)
    try:
        module = importlib.import_module(
            "extracted_content_pipeline.autonomous.tasks.competitive_intelligence"
        )
        assert hasattr(module, "run")
    finally:
        _evict(*_AFFECTED_MODULES)
