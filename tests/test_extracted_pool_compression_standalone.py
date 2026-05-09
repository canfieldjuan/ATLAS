"""Standalone-mode regression test for _b2b_pool_compression.

Pins the contract that
``extracted_content_pipeline.autonomous.tasks._b2b_pool_compression``
imports cleanly under ``EXTRACTED_PIPELINE_STANDALONE=1``. Pre-PR the
mirror failed with ``ModuleNotFoundError: No module named
'extracted_content_pipeline.autonomous.tasks._b2b_witnesses'``; the
standalone substrate added in PR-Decouple-PoolCompression-1 satisfies
that import.

Standalone contract is import-only. ``build_vendor_witness_artifacts``
returns empty witness data (``([], {})``) under the shim -- consistent
with the rest of the standalone surface. See plans/PR-Decouple-
PoolCompression-1.md.
"""

from __future__ import annotations

import importlib
import sys
from typing import Iterator

import pytest


_STANDALONE_ENV_VAR = "EXTRACTED_PIPELINE_STANDALONE"

_AFFECTED_MODULES = (
    "extracted_content_pipeline.autonomous.tasks._b2b_witnesses",
    "extracted_content_pipeline.autonomous.tasks._b2b_pool_compression",
)


def _evict(*module_names: str) -> None:
    for name in module_names:
        sys.modules.pop(name, None)


@pytest.fixture
def standalone_witnesses(monkeypatch) -> Iterator[object]:
    """Fresh import of the witnesses shim under
    ``EXTRACTED_PIPELINE_STANDALONE=1``."""
    monkeypatch.setenv(_STANDALONE_ENV_VAR, "1")
    _evict(*_AFFECTED_MODULES)
    try:
        module = importlib.import_module(
            "extracted_content_pipeline.autonomous.tasks._b2b_witnesses"
        )
        yield module
    finally:
        _evict(*_AFFECTED_MODULES)


def test_witnesses_shim_exports_required_symbol(standalone_witnesses):
    assert hasattr(standalone_witnesses, "build_vendor_witness_artifacts")


def test_build_vendor_witness_artifacts_returns_empty_pair(standalone_witnesses):
    build = standalone_witnesses.build_vendor_witness_artifacts
    pack, packets = build("Acme Corp", [{"text": "ignored"}])
    assert pack == []
    assert packets == {}


def test_build_vendor_witness_artifacts_accepts_real_signature_kwargs(
    standalone_witnesses,
):
    """The atlas_brain implementation accepts 8 keyword args with defaults.
    The shim's ``**_kwargs`` should absorb all of them without TypeError."""
    build = standalone_witnesses.build_vendor_witness_artifacts
    pack, packets = build(
        "Acme Corp",
        [],
        max_witnesses=12,
        min_specificity_score=0.5,
        fallback_min_witnesses=4,
        generic_patterns=("foo",),
        concrete_patterns=("bar",),
        short_excerpt_chars=120,
        long_excerpt_chars=480,
        specificity_weights={"a": 1.0},
    )
    assert pack == []
    assert packets == {}


def test_pool_compression_mirror_imports_under_standalone(monkeypatch):
    """The load-bearing assertion for this PR.

    Pre-PR this raised ``ModuleNotFoundError: No module named
    'extracted_content_pipeline.autonomous.tasks._b2b_witnesses'``.
    """
    monkeypatch.setenv(_STANDALONE_ENV_VAR, "1")
    _evict(*_AFFECTED_MODULES)
    try:
        module = importlib.import_module(
            "extracted_content_pipeline.autonomous.tasks._b2b_pool_compression"
        )
        # _b2b_pool_compression is a helper module; just confirm it loaded.
        assert module is not None
    finally:
        _evict(*_AFFECTED_MODULES)
