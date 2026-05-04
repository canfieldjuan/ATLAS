"""Tests for the evidence_hash consolidation (PR-A5c).

Before this PR, two functions computed the same SHA-256[:16] over a
JSON-canonicalised dict:

  * ``atlas_brain.reasoning.semantic_cache.compute_evidence_hash``
  * ``atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis.compute_cross_vendor_evidence_hash``

Their bodies were byte-identical. This PR keeps both names (Rule 14
preservation) but makes them aliases of a single implementation in
``semantic_cache``. These tests pin the consolidation:

  * the two names must resolve to the SAME function object, not just
    return-equal outputs (so a future drift between the two
    implementations cannot reintroduce);
  * representative inputs must produce stable hashes that match what
    the legacy duplicate produced (regression guard against an
    accidental canonicalisation change).
"""

from __future__ import annotations

import hashlib
import json

import pytest

from atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis import (
    compute_cross_vendor_evidence_hash,
)
from atlas_brain.reasoning.semantic_cache import compute_evidence_hash


# ---- Single-owner identity ----


def test_two_names_resolve_to_same_function_object():
    """The legacy public name MUST be a binding to the canonical
    implementation, not a wrapper or copy. ``is`` (identity) catches
    a future re-introduction of the duplicate at module-import time.
    """
    assert compute_cross_vendor_evidence_hash is compute_evidence_hash


# ---- Hash stability (regression guards) ----


def _expected(payload: dict) -> str:
    """Recompute the canonical hash inline so the expected value is
    derived from the same algorithm the production code uses, but
    expressed as a single explicit pipeline. If a future refactor
    accidentally changes the canonicalisation, this independent
    re-computation forces the test to flag it.
    """
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def test_hash_matches_canonical_pipeline_on_simple_dict():
    payload = {"vendor": "acme", "score": 7}
    assert compute_evidence_hash(payload) == _expected(payload)


def test_hash_is_16_hex_chars():
    h = compute_evidence_hash({"a": 1})
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


def test_hash_is_key_order_invariant():
    a = {"x": 1, "y": 2, "z": [3, 4]}
    b = {"z": [3, 4], "y": 2, "x": 1}
    assert compute_evidence_hash(a) == compute_evidence_hash(b)


def test_hash_changes_with_value_change():
    base = {"a": 1, "b": 2}
    bumped = {"a": 1, "b": 3}
    assert compute_evidence_hash(base) != compute_evidence_hash(bumped)


def test_hash_changes_with_nested_change():
    base = {"layers": [{"name": "a"}, {"name": "b"}]}
    swapped = {"layers": [{"name": "b"}, {"name": "a"}]}
    # Order in lists IS significant (lists aren't sorted)
    assert compute_evidence_hash(base) != compute_evidence_hash(swapped)


def test_hash_handles_non_json_native_values_via_default_str():
    from datetime import date
    payload = {"as_of": date(2026, 5, 4)}
    h = compute_evidence_hash(payload)
    assert len(h) == 16
    # Same as: '{"as_of":"2026-05-04"}' hashed
    assert h == _expected(payload)


def test_hash_empty_dict_returns_known_prefix():
    """Sanity guard against an accidental separator/sort-key change:
    the empty dict has a known canonical SHA-256[:16] = "44136fa355b3678a".
    """
    assert compute_evidence_hash({}) == "44136fa355b3678a"


# ---- Cross-call identity ----


def test_alias_produces_same_hash_as_canonical():
    payload = {"packet": {"v": 1}, "as_of": "2026-05-04"}
    assert (
        compute_cross_vendor_evidence_hash(payload)
        == compute_evidence_hash(payload)
    )


def test_caller_module_re_export_stays_resolvable():
    """``b2b_reasoning_synthesis.py`` imports the public name from
    ``_b2b_cross_vendor_synthesis``. The lazy import inside that
    function must keep working after the consolidation -- guards
    against a missing __all__ or a wrong import path.
    """
    from atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis import (
        compute_cross_vendor_evidence_hash as resolved,
    )
    assert resolved is compute_evidence_hash


# ---- __all__ contract ----


def test_module_exports_legacy_public_name():
    """``__all__`` on the cross-vendor synthesis module must list the
    public name so ``from ... import *`` keeps working for any
    downstream consumer that depended on the legacy export.
    """
    import atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis as mod
    assert "compute_cross_vendor_evidence_hash" in mod.__all__
